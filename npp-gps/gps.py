# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars>=1.36.1",
#     "geopy",
#     "scipy",
#     "gpxpy",
#     "fitparse",
# ]
# ///


import logging
import math
import threading
import time
import urllib.request
from collections import OrderedDict
from enum import Enum, Flag, auto
from pathlib import Path

import numpy as np
import polars as pl
from geopy.distance import geodesic
from pupil_labs.neon_player import Plugin, action
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QDockWidget, QWidget
from qt_property_widgets.utilities import action_params, property_params
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class ModifyDirection(Flag):
    NONE = 0
    TOP = auto()
    RIGHT = auto()
    BOTTOM = auto()
    LEFT = auto()
    MOVE = auto()


class MapStyle(Enum):
    VIDEO_GAME_01 = "Video Game 01"
    GOOGLE_MAPS = "Google Maps"
    WAZE = "Waze"
    GTA_V = "GTA V"


class MapShape(Enum):
    CIRCLE = "Circle"
    SQUARE = "Square"
    SQUIRCLE = "Squircle"


class IMUTransforms:
    @staticmethod
    def spherical_to_cartesian_scene(
        elevations: np.ndarray, azimuths: np.ndarray
    ) -> np.ndarray:
        e_rad, a_rad = np.deg2rad(elevations), np.deg2rad(azimuths)
        e_rad += np.pi / 2
        a_rad *= -1.0
        a_rad += np.pi / 2
        return np.array([
            np.sin(e_rad) * np.cos(a_rad),
            np.cos(e_rad),
            np.sin(e_rad) * np.sin(a_rad),
        ]).T

    @staticmethod
    def transform_imu_to_world(
        imu_coords: np.ndarray, quaternions: np.ndarray
    ) -> np.ndarray:
        mats = R.from_quat(quaternions, scalar_first=True).as_matrix()
        if np.ndim(imu_coords) == 1:
            return mats @ imu_coords
        return np.array([m @ c for m, c in zip(mats, imu_coords, strict=False)])

    @staticmethod
    def transform_scene_to_imu(coords: np.ndarray) -> np.ndarray:
        rot_diff = np.deg2rad(-102)
        s2i = np.array([
            [1.0, 0, 0],
            [0, np.cos(rot_diff), -np.sin(rot_diff)],
            [0, np.sin(rot_diff), np.cos(rot_diff)],
        ])
        return (s2i @ coords.T).T + np.array([0.0, -1.3, -6.62])

    @staticmethod
    def cartesian_to_spherical_world(p3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y, z = p3d[:, 0], p3d[:, 1], p3d[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2 + 1e-12)
        ele = -(np.arccos(z / r) - np.pi / 2)
        azi = np.arctan2(y, x) - np.pi / 2
        azi[azi < -np.pi] += 2 * np.pi
        azi[azi > np.pi] -= 2 * np.pi
        return np.rad2deg(ele), np.rad2deg(azi)


class GPSDataLoader:
    @staticmethod
    def load(rec_dir: Path, cache_dir: Path) -> pl.DataFrame | None:
        cache_file = cache_dir / "gps_data_cached.feather"
        if cache_file.exists():
            try:
                df = pl.read_feather(cache_file)
                logger.info(f"GPS Plugin: Loaded fast from cache {cache_file}")
                return df
            except Exception as e:
                logger.warning(f"GPS Plugin: Cache corrupted, rebuilding... {e}")

        df = None
        csv_files = list(rec_dir.glob("gps_*.csv"))
        gpx_files = list(rec_dir.glob("*.gpx"))
        fit_files = list(rec_dir.glob("*.fit"))

        if csv_files:
            df = GPSDataLoader._load_csv(csv_files[0])
        elif gpx_files:
            df = GPSDataLoader._load_gpx(gpx_files[0])
        elif fit_files:
            df = GPSDataLoader._load_fit(fit_files[0])

        if df is not None and not df.is_empty():
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                df.write_feather(cache_file)
                logger.info(f"GPS Plugin: Cached parsed data to {cache_file}")
            except Exception as e:
                logger.warning(f"GPS Plugin: Failed to save cache: {e}")

        return df

    @staticmethod
    def _load_csv(path: Path) -> pl.DataFrame | None:
        df = pl.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df = df.drop_nulls(subset=["latitude", "longitude"])
        if df.is_empty():
            return None
        return df.sort("timestamp [ns]")

    @staticmethod
    def _load_gpx(path: Path) -> pl.DataFrame | None:
        try:
            import gpxpy

            with open(path) as gpx_file:
                gpx = gpxpy.parse(gpx_file)
            data = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        if point.time:
                            ts_ns = int(point.time.timestamp() * 1e9)
                            hr = None
                            for ext in point.extensions:
                                if len(ext) > 0:
                                    for child in ext:
                                        if child.tag.endswith("hr"):
                                            hr = float(child.text)
                                elif ext.tag.endswith("hr"):
                                    hr = float(ext.text)
                            data.append({
                                "timestamp [ns]": ts_ns,
                                "latitude": point.latitude,
                                "longitude": point.longitude,
                                "elevation": point.elevation,
                                "heart_rate": hr,
                            })
            if not data:
                return None
            df = pl.DataFrame(data)
            return df.sort("timestamp [ns]")
        except ImportError:
            logger.warning("gpxpy not installed. Cannot parse GPX.")
            return None
        except Exception as e:
            logger.exception(f"Error parsing GPX: {e}")
            return None

    @staticmethod
    def _load_fit(path: Path) -> pl.DataFrame | None:
        try:
            from fitparse import FitFile

            fitfile = FitFile(str(path))
            data = []
            for record in fitfile.get_messages("record"):
                record_data = {}
                for record_data_entry in record:
                    record_data[record_data_entry.name] = record_data_entry.value

                if (
                    "position_lat" in record_data
                    and "position_long" in record_data
                    and "timestamp" in record_data
                ):
                    lat = (
                        record_data["position_lat"] * (180.0 / 2**31)
                        if record_data["position_lat"]
                        else None
                    )
                    lon = (
                        record_data["position_long"] * (180.0 / 2**31)
                        if record_data["position_long"]
                        else None
                    )
                    ts = record_data["timestamp"]
                    hr = record_data.get("heart_rate")
                    ele = record_data.get(
                        "enhanced_altitude", record_data.get("altitude")
                    )
                    if lat is not None and lon is not None and ts is not None:
                        ts_ns = int(ts.timestamp() * 1e9)
                        data.append({
                            "timestamp [ns]": ts_ns,
                            "latitude": lat,
                            "longitude": lon,
                            "elevation": ele,
                            "heart_rate": hr,
                        })
            if not data:
                return None
            df = pl.DataFrame(data)
            return df.sort("timestamp [ns]")
        except ImportError:
            logger.warning("fitparse not installed. Cannot parse FIT.")
            return None
        except Exception as e:
            logger.exception(f"Error parsing FIT: {e}")
            return None


class TileManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "tiles"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pixmap_cache: OrderedDict[tuple[int, int, int, str], QPixmap] = (
            OrderedDict()
        )
        self._loading: set[tuple[int, int, int, str]] = set()
        self._max_cache = 200

    def get_tile(self, x: int, y: int, z: int, style: MapStyle) -> QPixmap | None:
        key = (x, y, z, style.value)
        if key in self._pixmap_cache:
            self._pixmap_cache.move_to_end(key)
            return self._pixmap_cache[key]

        path = self.cache_dir / f"{style.value.replace(' ', '_')}_{z}_{x}_{y}.png"
        if path.exists():
            img = QImage(str(path))
            if not img.isNull():
                pix = QPixmap.fromImage(img)
                self._pixmap_cache[key] = pix
                if len(self._pixmap_cache) > self._max_cache:
                    self._pixmap_cache.popitem(last=False)
                return pix
            else:
                path.unlink()

        if key not in self._loading:
            self._loading.add(key)
            threading.Thread(
                target=self._download_tile, args=(x, y, z, style), daemon=True
            ).start()
        return None

    def _download_tile(self, x: int, y: int, z: int, style: MapStyle):
        if style == MapStyle.VIDEO_GAME_01:
            url = f"https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
        elif style == MapStyle.GOOGLE_MAPS:
            url = f"https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"
        elif style == MapStyle.WAZE:
            url = f"https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
        else:
            url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"

        path = self.cache_dir / f"{style.value.replace(' ', '_')}_{z}_{x}_{y}.png"
        try:
            req = urllib.request.Request(  # ruff: ignore[suspicious-url-open-usage]
                url,
                headers={"User-Agent": "NeonPlayer-GPS-Plugin/2.0 (pupil-labs.com)"},
            )
            with urllib.request.urlopen(req) as resp:  # ruff: ignore[suspicious-url-open-usage]
                data = resp.read()
                Path(path).write_bytes(data)
        except Exception as e:
            logger.debug(f"Tile download failed for {url}: {e}")
        finally:
            self._loading.discard((x, y, z, style.value))


class NativeMapView(QWidget):
    def __init__(self, plugin, parent=None):
        super().__init__(parent)
        self.plugin = plugin
        self.setMouseTracking(True)
        self.center_lat = 0.0
        self.center_lon = 0.0
        self.zoom = 15.0
        self.current_idx = 0
        self.gazi = 0.0
        self._dragging = False
        self._last_mouse_pos = None
        self._click_start_pos = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._check_loading)
        self.timer.start(100)

    def _check_loading(self):
        if self.plugin._tm and len(self.plugin._tm._loading) > 0:
            self.update()

    def load_map(self):
        df = self.plugin.gps_data
        if df is None or df.is_empty():
            return
        self.center_lat = float(np.mean(df["latitude"].to_numpy()))
        self.center_lon = float(np.mean(df["longitude"].to_numpy()))
        self.update()

    def update_wearer_position(self, lat: float, lon: float, idx: int, gazi: float):
        self.current_idx = idx
        self.gazi = gazi
        self.update()

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        if angle > 0:
            self.zoom += 0.5
        else:
            self.zoom -= 0.5
        self.zoom = max(1.0, min(19.0, self.zoom))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_mouse_pos = event.pos()
            self._click_start_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            if self._click_start_pos is not None:
                dist = (event.pos() - self._click_start_pos).manhattanLength()
                if dist < 5:
                    z = int(self.zoom)
                    n = 2.0**z
                    px_deg = 360.0 / (256 * n)
                    cx = self.width() / 2
                    cy = self.height() / 2
                    dx = event.pos().x() - cx
                    dy = event.pos().y() - cy
                    lon = self.center_lon + dx * px_deg
                    lat = self.center_lat - dy * px_deg
                    self.plugin.seek_to_nearest_gps(lat, lon)
            self._click_start_pos = None

    def mouseMoveEvent(self, event):
        if self._dragging and self._last_mouse_pos is not None:
            dx = event.pos().x() - self._last_mouse_pos.x()
            dy = event.pos().y() - self._last_mouse_pos.y()
            self._last_mouse_pos = event.pos()

            z = int(self.zoom)
            n = 2.0**z
            px_deg = 360.0 / (256 * n)

            self.center_lon -= dx * px_deg
            self.center_lat += dy * px_deg
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor("#1e1e1e"))

            if self.plugin.gps_data is None:
                return

            clat = self.center_lat
            clon = self.center_lon
            z = int(self.zoom)
            n = 2.0**z
            xt = (clon + 180.0) / 360.0 * n
            yt = (1.0 - math.asinh(math.tan(math.radians(clat))) / math.pi) / 2.0 * n

            tile_x, tile_y = int(xt), int(yt)
            off_x, off_y = (xt - tile_x) * 256, (yt - tile_y) * 256

            cx = self.width() / 2
            cy = self.height() / 2

            painter.translate(cx, cy)

            px_deg = 360.0 / (256 * n)

            tm = self.plugin._tm
            if tm:
                tiles_x = (self.width() // 256) // 2 + 2
                tiles_y = (self.height() // 256) // 2 + 2
                for dx in range(-tiles_x, tiles_x + 1):
                    for dy in range(-tiles_y, tiles_y + 1):
                        pix = tm.get_tile(
                            tile_x + dx, tile_y + dy, z, self.plugin._map_style
                        )
                        if pix:
                            painter.drawPixmap(
                                QPointF((dx * 256) - off_x, (dy * 256) - off_y), pix
                            )

            lats = self.plugin.gps_data["latitude"].to_numpy()
            lons = self.plugin.gps_data["longitude"].to_numpy()

            current_idx = self.current_idx

            poly_past = QPolygonF()
            poly_future = QPolygonF()

            step = max(1, len(lats) // 1000)
            for i in range(0, current_idx, step):
                poly_past.append(
                    QPointF((lons[i] - clon) / px_deg, -(lats[i] - clat) / px_deg)
                )
            if current_idx < len(lats):
                poly_past.append(
                    QPointF(
                        (lons[current_idx] - clon) / px_deg,
                        -(lats[current_idx] - clat) / px_deg,
                    )
                )

            for i in range(current_idx, len(lats), step):
                poly_future.append(
                    QPointF((lons[i] - clon) / px_deg, -(lats[i] - clat) / px_deg)
                )
            if len(lats) > 0:
                poly_future.append(
                    QPointF((lons[-1] - clon) / px_deg, -(lats[-1] - clat) / px_deg)
                )

            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            painter.setPen(QPen(QColor(150, 150, 150, 150), 3, Qt.PenStyle.DashLine))
            painter.drawPolyline(poly_future)

            painter.setPen(QPen(QColor(26, 115, 232, 255), 4, Qt.PenStyle.SolidLine))
            painter.drawPolyline(poly_past)

            events_plugin = self.plugin.app.plugins_by_class.get("EventsPlugin")
            if events_plugin and events_plugin.events:
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.setBrush(QBrush(QColor(255, 50, 50)))
                for timestamps in events_plugin.events.values():
                    for ts in timestamps:
                        pos_evt = self.plugin.get_pos_at_time(ts)
                        if pos_evt and not np.isnan(pos_evt[0]):
                            px = (pos_evt[1] - clon) / px_deg
                            py = -(pos_evt[0] - clat) / px_deg
                            painter.drawEllipse(QPointF(px, py), 4, 4)

            if len(lats) > 0:
                wlat, wlon = lats[current_idx], lons[current_idx]
                px = (wlon - clon) / px_deg
                py = -(wlat - clat) / px_deg

                painter.translate(px, py)
                painter.rotate(-self.gazi)

                w = 10
                poly = QPolygonF([
                    QPointF(-w / 1.5, w / 1.5),
                    QPointF(w / 1.5, w / 1.5),
                    QPointF(0, -w),
                ])
                painter.setPen(QPen(Qt.GlobalColor.white, 2))
                painter.setBrush(QBrush(QColor(51, 136, 255)))
                painter.drawPolygon(poly)

                painter.rotate(self.gazi)
                painter.translate(-px, -py)
        finally:
            painter.end()


class GPSMapDock(QDockWidget):
    def __init__(self, plugin, parent=None):
        super().__init__("GPS Tracking - Full Map", parent)
        self.plugin = plugin
        self.map_view = NativeMapView(plugin, self)
        self.setWidget(self.map_view)
        self.load_map()

    def load_map(self):
        self.map_view.load_map()

    def update_wearer_position(self, lat: float, lon: float, idx: int, gazi: float):
        self.map_view.update_wearer_position(lat, lon, idx, gazi)


class GPSPlugin(Plugin):
    label = "GPS Plugin"

    def __init__(self):
        super().__init__()
        self.gps_data: pl.DataFrame | None = None
        self._raw_gps_data: pl.DataFrame | None = None
        self._timeline_rows = []
        self._tm: TileManager | None = None

        self._time_offset_sec = 0.0
        self._dock: GPSMapDock | None = None

        self._show_minimap = True
        self._minimap_zoom = 18
        self._offset_x = 0.02
        self._offset_y = 0.02
        self._scale = 1.0
        self._minimap_anchor_right = True
        self._minimap_anchor_bottom = True
        self._lock = False
        self._map_style = MapStyle.VIDEO_GAME_01
        self._map_shape = MapShape.CIRCLE
        self._alpha = 0.85
        self._show_full_path = True

        self._mouse_mode = ModifyDirection.NONE
        self._drag_start_pos = None
        self._start_geometry = QRectF()
        self._last_click_time = 0.0

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        if self._dock is not None:
            try:
                self.app.main_window.removeDockWidget(self._dock)
                self._dock.close()
                self._dock.deleteLater()
            except Exception:
                pass
            self._dock = None

        self.gps_data = None
        self._raw_gps_data = None

        tl = self.get_timeline()
        if tl:
            for r in self._timeline_rows:
                tl.remove_timeline_plot(r)
        self._timeline_rows.clear()

        self._tm = TileManager(self.get_cache_path())
        self.load_gps_data()

        vw = self.app.main_window.video_widget
        try:
            vw.mouse_pressed.disconnect(self.on_mouse_pressed)
            vw.mouse_moved.disconnect(self.on_mouse_moved)
            vw.mouse_released.disconnect(self.on_mouse_released)
        except Exception:  # ruff: ignore[blind-except, try-except-pass]
            pass

        vw.mouse_pressed.connect(self.on_mouse_pressed)
        vw.mouse_moved.connect(self.on_mouse_moved)
        vw.mouse_released.connect(self.on_mouse_released)

    def load_gps_data(self):
        rd = getattr(
            self.recording, "rec_dir", getattr(self.recording, "_rec_dir", None)
        )
        if not rd:
            return

        cache_path = self.get_cache_path()
        df = GPSDataLoader.load(Path(rd), cache_path)
        if df is None:
            return

        fit_files = list(Path(rd).glob("*.fit"))
        gpx_files = list(Path(rd).glob("*.gpx"))
        self._is_absolute_time = len(fit_files) > 0 or len(gpx_files) > 0

        try:
            self._raw_gps_data = df
            self.apply_time_offset()
            logger.info(f"GPS Plugin: Loaded {len(self.gps_data)} points.")
        except Exception:
            logger.exception("GPS load error")

    def apply_time_offset(self):
        if self._raw_gps_data is None:
            return

        manual_offset_ns = int(self._time_offset_sec * 1e9)
        auto_offset_ns = 0

        if getattr(self, "_is_absolute_time", False):
            # Treat recording start as baseline
            try:
                rec_start_ns = int(self.recording.imu.time[0])
                # Treat GPS data relative to that
                gps_start_ns = int(self._raw_gps_data["timestamp [ns]"][0])
                auto_offset_ns = rec_start_ns - gps_start_ns
            except Exception as e:
                logger.exception(f"Baseline alignment failed: {e}")

        total_offset_ns = auto_offset_ns + manual_offset_ns

        self.gps_data = self._raw_gps_data.with_columns(
            (pl.col("timestamp [ns]") + total_offset_ns).alias("timestamp [ns]")
        )
        self.recording.gps = self.gps_data
        self.calculate_metrics()
        self.update_timeline()
        self.changed.emit()

    def calculate_metrics(self):
        if self.gps_data is None or self.gps_data.is_empty():
            return

        lats = self.gps_data["latitude"].to_numpy()
        lons = self.gps_data["longitude"].to_numpy()
        tss = self.gps_data["timestamp [ns]"].to_numpy()

        dists = np.zeros(len(lats))
        brngs = np.zeros(len(lats))

        for i in range(1, len(lats)):
            dists[i] = geodesic((lats[i - 1], lons[i - 1]), (lats[i], lons[i])).meters
            dl = np.radians(lons[i] - lons[i - 1])
            y = np.sin(dl) * np.cos(np.radians(lats[i]))
            x = np.cos(np.radians(lats[i - 1])) * np.sin(np.radians(lats[i])) - np.sin(
                np.radians(lats[i - 1])
            ) * np.cos(np.radians(lats[i])) * np.cos(dl)
            brngs[i] = (np.degrees(np.arctan2(y, x)) + 360) % 360

        dt = np.diff(tss) / 1e9
        dt = np.concatenate(([1.0], dt))
        dt[dt <= 0] = 1.0

        vel = dists / dt
        acc = np.concatenate(([0.0], np.diff(vel) / dt[1:]))

        cols_to_add = [
            pl.Series("velocity", vel),
            pl.Series("acceleration", acc),
            pl.Series("bearing", brngs),
        ]
        self.gps_data = self.gps_data.with_columns(cols_to_add)

    def get_pos_at_time(self, time_ns: int) -> tuple[float, float, int] | None:
        if self.gps_data is None or self.gps_data.is_empty():
            return None
        tss = self.gps_data["timestamp [ns]"].to_numpy()
        idx = int(np.searchsorted(tss, time_ns))

        if idx == 0:
            return (
                float(self.gps_data["latitude"][0]),
                float(self.gps_data["longitude"][0]),
                0,
            )
        if idx >= len(tss):
            return (
                float(self.gps_data["latitude"][-1]),
                float(self.gps_data["longitude"][-1]),
                len(tss) - 1,
            )

        t0, t1 = int(tss[idx - 1]), int(tss[idx])
        f = (time_ns - t0) / (t1 - t0)
        l0, l1 = (
            float(self.gps_data["latitude"][idx - 1]),
            float(self.gps_data["latitude"][idx]),
        )
        o0, o1 = (
            float(self.gps_data["longitude"][idx - 1]),
            float(self.gps_data["longitude"][idx]),
        )
        return l0 + f * (l1 - l0), o0 + f * (o1 - o0), idx

    def get_gaze_world_azi_at_time(self, t_ns: int) -> float:
        gs = self.recording.gaze.sample([t_ns])
        ims = self.recording.imu.sample([t_ns])
        if not gs or not ims or not gs[0] or not ims[0]:
            return np.nan
        try:
            g, imu = gs[0], ims[0]
            azi = getattr(g, "azimuth", getattr(g, "azimuth_deg", 0.0))
            ele = getattr(g, "elevation", getattr(g, "elevation_deg", 0.0))
            cs = IMUTransforms.spherical_to_cartesian_scene(ele, azi)
            ci = IMUTransforms.transform_scene_to_imu(cs)
            q = (
                imu.quaternion
                if hasattr(imu, "quaternion")
                else [
                    imu.quaternion_w,
                    imu.quaternion_x,
                    imu.quaternion_y,
                    imu.quaternion_z,
                ]
            )
            cw = IMUTransforms.transform_imu_to_world(ci, np.array([q]))[0]
            _, ga = IMUTransforms.cartesian_to_spherical_world(np.array([cw]))
            return ga[0]
        except Exception:
            return np.nan

    def seek_to_nearest_gps(self, lat: float, lon: float):
        if self.gps_data is None or self.gps_data.is_empty():
            return
        lats = self.gps_data["latitude"].to_numpy()
        lons = self.gps_data["longitude"].to_numpy()

        dists = (lats - lat) ** 2 + (lons - lon) ** 2
        idx = int(np.argmin(dists))
        ts_ns = int(self.gps_data["timestamp [ns]"][idx])

        if self.app:
            self.app.seek_to(ts_ns)

    def on_disabled(self) -> None:
        if getattr(self, "_dock", None) is not None:
            try:
                self.app.main_window.removeDockWidget(self._dock)
                self._dock.close()
                self._dock.deleteLater()
            except Exception:  # ruff: ignore[blind-except, try-except-pass]
                pass
            self._dock = None

        tl = self.get_timeline()
        if not tl:
            return
        for r in self._timeline_rows:
            tl.remove_timeline_plot(r)
        self._timeline_rows.clear()

    def update_timeline(self):
        tl = self.get_timeline()
        if not tl or self.gps_data is None:
            return
        for r in self._timeline_rows:
            tl.remove_timeline_plot(r)
        self._timeline_rows.clear()

        # Group Kinematics Together
        kin_name = "GPS Kinematics"
        self._timeline_rows.append(kin_name)

        tss = self.gps_data["timestamp [ns]"].to_numpy().astype(np.float64)
        vel = self.gps_data["velocity"].to_numpy().astype(np.float64)
        acc = self.gps_data["acceleration"].to_numpy().astype(np.float64)

        tl.add_timeline_line(kin_name, np.column_stack((tss, vel)), "Velocity (m/s)")
        tl.add_timeline_line(
            kin_name, np.column_stack((tss, acc)), "Acceleration (m/s²)"
        )

        if "elevation" in self.gps_data.columns:
            ele_series = self.gps_data["elevation"]
            if ele_series.null_count() < len(ele_series):
                ele_name = "GPS Elevation (m)"
                self._timeline_rows.append(ele_name)
                df_ele = self.gps_data.drop_nulls(subset=["elevation"]).sort(
                    "timestamp [ns]"
                )
                tss_ele = df_ele["timestamp [ns]"].to_numpy().astype(np.float64)
                ele = df_ele["elevation"].to_numpy().astype(np.float64)

                tl.add_timeline_line(
                    ele_name, np.column_stack((tss_ele, ele)), "Elevation"
                )

                p_ele = tl.get_timeline_plot(ele_name)
                if p_ele:
                    import pyqtgraph as pg

                    for item in p_ele.items:
                        if isinstance(item, (pg.PlotDataItem, pg.PlotCurveItem)):
                            item.setPen(pg.mkPen("g", width=1.5))

        if "heart_rate" in self.gps_data.columns:
            hr_series = self.gps_data["heart_rate"]
            if hr_series.null_count() < len(hr_series):
                hr_name = "GPS Heart Rate (bpm)"
                self._timeline_rows.append(hr_name)
                df_hr = self.gps_data.drop_nulls(subset=["heart_rate"]).sort(
                    "timestamp [ns]"
                )
                tss_hr = df_hr["timestamp [ns]"].to_numpy().astype(np.float64)
                hr = df_hr["heart_rate"].to_numpy().astype(np.float64)

                tl.add_timeline_line(
                    hr_name, np.column_stack((tss_hr, hr)), "Heart Rate"
                )

                p_hr = tl.get_timeline_plot(hr_name)
                if p_hr:
                    import pyqtgraph as pg

                    for item in p_hr.items:
                        if isinstance(item, (pg.PlotDataItem, pg.PlotCurveItem)):
                            item.setPen(pg.mkPen("r", width=1.5))

    def get_minimap_rect(self) -> QRectF:
        sw, sh = self.recording.scene.width, self.recording.scene.height
        base_size = 300 * self._scale
        mx = (
            (sw - base_size - self._offset_x * sw)
            if self._minimap_anchor_right
            else (self._offset_x * sw)
        )
        my = (
            (sh - base_size - self._offset_y * sh)
            if self._minimap_anchor_bottom
            else (self._offset_y * sh)
        )
        return QRectF(mx, my, base_size, base_size)

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        try:
            self._render_impl(painter, time_in_recording)
        except Exception:
            import traceback

            traceback.print_exc()

    def _render_impl(self, painter: QPainter, time_ns: int) -> None:
        if self.gps_data is None or self.gps_data.is_empty() or not self._show_minimap:
            return

        pos_data = self.get_pos_at_time(time_ns)
        if not pos_data or np.isnan(pos_data[0]):
            return

        clat, clon, current_idx = pos_data

        gazi = self.get_gaze_world_azi_at_time(time_ns)
        gazi = 0 if np.isnan(gazi) else gazi

        if self._dock is not None:
            try:
                if self._dock.isVisible():
                    self._dock.update_wearer_position(clat, clon, current_idx, gazi)
            except RuntimeError:
                self._dock = None

        rect = self.get_minimap_rect()
        cx, cy = rect.center().x(), rect.center().y()

        painter.save()
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )

        if self._map_style == MapStyle.VIDEO_GAME_01:
            bg_color = QColor(20, 30, 40)
        elif self._map_style == MapStyle.GOOGLE_MAPS:
            bg_color = QColor(240, 238, 233)
        else:  # WAZE
            bg_color = QColor(240, 240, 240)

        path = QPainterPath()
        if self._map_shape == MapShape.CIRCLE:
            path.addEllipse(rect)
        elif self._map_shape == MapShape.SQUARE:
            path.addRect(rect)
        elif self._map_shape == MapShape.SQUIRCLE:
            path.addRoundedRect(rect, rect.width() * 0.2, rect.height() * 0.2)

        painter.setClipPath(path)
        painter.setOpacity(self._alpha)

        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)

        z = self._minimap_zoom
        n = 2.0**z
        xt = (clon + 180.0) / 360.0 * n
        yt = (1.0 - math.asinh(math.tan(math.radians(clat))) / math.pi) / 2.0 * n
        tile_x, tile_y = int(xt), int(yt)
        off_x, off_y = (xt - tile_x) * 256, (yt - tile_y) * 256
        px_deg = 360.0 / (256 * n)

        painter.translate(cx, cy)

        if not self._lock:
            painter.rotate(-gazi)

        if self._tm:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    pix = self._tm.get_tile(
                        tile_x + dx, tile_y + dy, z, self._map_style
                    )
                    if pix:
                        painter.drawPixmap(
                            int((dx * 256) - off_x), int((dy * 256) - off_y), pix
                        )

        painter.setOpacity(1.0)

        lats = self.gps_data["latitude"].to_numpy()
        lons = self.gps_data["longitude"].to_numpy()

        if self._map_style == MapStyle.VIDEO_GAME_01:
            path_color = QColor(50, 150, 255, 255)
            path_width = 5
            future_color = QColor(150, 150, 150, 150)
        elif self._map_style == MapStyle.GOOGLE_MAPS:
            path_color = QColor(26, 115, 232, 255)
            path_width = 6
            future_color = QColor(150, 150, 150, 150)
        elif self._map_style == MapStyle.GTA_V:
            path_color = QColor(100, 255, 100, 255)  # Solid Neon Green
            path_width = 6
            future_color = QColor(100, 255, 100, 150)
        else:  # WAZE
            path_color = QColor(160, 90, 255, 255)
            path_width = 6
            future_color = QColor(200, 200, 200, 150)

        if self._show_full_path:
            painter.setPen(
                QPen(
                    future_color,
                    path_width - 2,
                    Qt.PenStyle.DashLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            poly_future = QPolygonF()
            step_future = max(1, (len(lats) - current_idx) // 200)
            for i in range(current_idx, len(lats), step_future):
                poly_future.append(
                    QPointF((lons[i] - clon) / px_deg, -(lats[i] - clat) / px_deg)
                )
            if len(lats) > 0:
                poly_future.append(
                    QPointF((lons[-1] - clon) / px_deg, -(lats[-1] - clat) / px_deg)
                )
            painter.drawPolyline(poly_future)

        painter.setPen(
            QPen(path_color, path_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )

        poly_past = QPolygonF()
        step_past = max(1, current_idx // 200)
        start_idx = 0 if self._show_full_path else max(0, current_idx - 300)

        for i in range(start_idx, current_idx, step_past):
            poly_past.append(
                QPointF((lons[i] - clon) / px_deg, -(lats[i] - clat) / px_deg)
            )
        poly_past.append(QPointF(0, 0))
        painter.drawPolyline(poly_past)

        events_plugin = Plugin.get_instance_by_name("EventsPlugin")
        if events_plugin and events_plugin.events:
            painter.setPen(QPen(Qt.GlobalColor.white, 1.5))
            painter.setBrush(QBrush(QColor(255, 50, 50)))  # Red markers
            for timestamps in events_plugin.events.values():
                for ts in timestamps:
                    pos_evt = self.get_pos_at_time(ts)
                    if pos_evt and pos_evt[0] is not None and not np.isnan(pos_evt[0]):
                        if (
                            abs(pos_evt[0] - clat) < self._minimap_zoom * 0.001
                            and abs(pos_evt[1] - clon) < self._minimap_zoom * 0.001
                        ):
                            px = (pos_evt[1] - clon) / px_deg
                            py = -(pos_evt[0] - clat) / px_deg
                            painter.drawEllipse(QPointF(px, py), 5, 5)

        painter.restore()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._mouse_mode != ModifyDirection.NONE:
            painter.setPen(QPen(QColor("#6D7BE0"), 6))
            painter.drawPath(path)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._map_style == MapStyle.VIDEO_GAME_01:
            painter.setPen(QPen(QColor(40, 40, 40), 8))
            painter.drawPath(path)
            painter.setPen(QPen(QColor(220, 220, 220), 2))
            painter.drawPath(path)
        elif self._map_style == MapStyle.GTA_V:
            painter.setPen(QPen(QColor(10, 10, 10), 6))
            painter.drawPath(path)
            painter.setPen(QPen(QColor(100, 100, 100), 2))
            painter.drawPath(path)
        elif self._map_style == MapStyle.GOOGLE_MAPS:
            painter.setPen(QPen(QColor(200, 200, 200), 2))
            painter.drawPath(path)
        elif self._map_style == MapStyle.WAZE:
            painter.setPen(QPen(QColor(200, 200, 200), 4))
            painter.drawPath(path)

        painter.translate(cx, cy)

        if self._lock:
            painter.rotate(gazi)

        if (
            self._map_style == MapStyle.VIDEO_GAME_01
            or self._map_style == MapStyle.GTA_V
        ):
            arrow = QPolygonF([
                QPointF(0, -18),
                QPointF(12, 12),
                QPointF(0, 6),
                QPointF(-12, 12),
            ])
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 2.5))
            painter.drawPolygon(arrow)
        elif self._map_style == MapStyle.GOOGLE_MAPS:
            cone = QPainterPath()
            cone.moveTo(0, 0)
            cone.arcTo(-40, -40, 80, 80, 50, 80)
            cone.closeSubpath()
            painter.setBrush(QBrush(QColor(26, 115, 232, 100)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(cone)

            painter.setBrush(QBrush(QColor(26, 115, 232)))
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.drawEllipse(QPointF(-3.5, -3.5), 7, 7)
        elif self._map_style == MapStyle.WAZE:
            arrow = QPolygonF([
                QPointF(0, -14),
                QPointF(12, 10),
                QPointF(0, 4),
                QPointF(-12, 10),
            ])
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 150, 255), 3))
            painter.drawPolygon(arrow)

        painter.restore()

    def on_mouse_pressed(self, event):
        vw = self.app.main_window.video_widget
        try:
            pos = vw.map_point(event.position())
        except AttributeError:
            pos = vw.map_point(event.pos())

        rect = self.get_minimap_rect()

        if rect.contains(pos):
            now = time.time()
            if now - self._last_click_time < 0.3:
                self.show_full_map()
                self._last_click_time = 0
                return
            self._last_click_time = now

            self._drag_start_pos = pos
            self._start_geometry = rect
            self.on_hover(event)
            event.accept()

    def on_mouse_moved(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.on_drag(event)
        else:
            self.on_hover(event)

    def on_mouse_released(self, event):
        self._mouse_mode = ModifyDirection.NONE
        self._drag_start_pos = None
        self.app.main_window.video_widget.unsetCursor()
        self.changed.emit()

    def on_hover(self, event):
        vw = self.app.main_window.video_widget
        try:
            pos = vw.map_point(event.position())
        except AttributeError:
            pos = vw.map_point(event.pos())

        rect = self.get_minimap_rect()
        margin = 45

        if rect.contains(pos):
            self._mouse_mode = ModifyDirection.NONE
            if pos.x() > rect.right() - margin and pos.y() > rect.bottom() - margin:
                self._mouse_mode = ModifyDirection.BOTTOM | ModifyDirection.RIGHT
                vw.setCursor(Qt.CursorShape.SizeFDiagCursor)
            else:
                self._mouse_mode = ModifyDirection.MOVE
                vw.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self._mouse_mode = ModifyDirection.NONE
            vw.unsetCursor()
        vw.update()

    def on_drag(self, event):
        if self._mouse_mode == ModifyDirection.NONE or not self._drag_start_pos:
            return

        vw = self.app.main_window.video_widget
        try:
            mapped_pos = vw.map_point(event.position())
        except AttributeError:
            mapped_pos = vw.map_point(event.pos())

        dx = mapped_pos.x() - self._drag_start_pos.x()
        dy = mapped_pos.y() - self._drag_start_pos.y()

        sw, sh = self.recording.scene.width, self.recording.scene.height

        if self._mouse_mode == ModifyDirection.MOVE:
            if self._minimap_anchor_right:
                self._offset_x -= dx / sw
            else:
                self._offset_x += dx / sw

            if self._minimap_anchor_bottom:
                self._offset_y -= dy / sh
            else:
                self._offset_y += dy / sh

            self._drag_start_pos = mapped_pos
        else:
            new_size = self._start_geometry.width() + dx
            self._scale = max(0.3, new_size / 300.0)

        self.changed.emit()
        vw.update()

    @action
    @action_params(icon=QIcon.fromTheme("window-new"), compact=True)
    def show_full_map(self):
        if self.gps_data is not None:
            try:
                if self._dock is not None:
                    self._dock.show()
                    self._dock.raise_()
                    return
            except RuntimeError:
                self._dock = None

            self._dock = GPSMapDock(self, self.app.main_window)
            self._dock.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self.app.main_window.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea, self._dock
            )
            self._dock.show()

    @property
    @property_params(
        label="Fine-Tune Offset (sec)",
        min=-3600.0,
        max=3600.0,
        step=0.1,
        description="Only effective for .fit and .gpx absolute time files.",
    )
    def time_offset_sec(self) -> float:
        return self._time_offset_sec

    @time_offset_sec.setter
    def time_offset_sec(self, v: float):
        self._time_offset_sec = float(v)
        self.apply_time_offset()

    @property
    def show_minimap(self) -> bool:
        return self._show_minimap

    @show_minimap.setter
    def show_minimap(self, v: bool):
        self._show_minimap = bool(v)
        self.changed.emit()

    @property
    def map_style(self) -> MapStyle:
        return self._map_style

    @map_style.setter
    def map_style(self, v: MapStyle):
        self._map_style = v
        self.changed.emit()

    @property
    def map_shape(self) -> MapShape:
        return self._map_shape

    @map_shape.setter
    def map_shape(self, v: MapShape):
        self._map_shape = v
        self.changed.emit()

    @property
    @property_params(
        min=1,
        max=19,
        step=1,
        description="OSM limits standard tiles to 19.",
    )
    def zoom(self) -> int:
        return self._minimap_zoom

    @zoom.setter
    def zoom(self, v: int):
        self._minimap_zoom = int(v)
        self.changed.emit()

    @property
    @property_params(label="Transparency (Alpha)", min=0.1, max=1.0, step=0.05)
    def opacity(self) -> float:
        return self._alpha

    @opacity.setter
    def opacity(self, v: float):
        self._alpha = float(v)
        self.changed.emit()

    @property
    @property_params(min=0.1, max=5.0, step=0.1)
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, v: float):
        self._scale = float(v)
        self.changed.emit()

    @property
    @property_params(label="Lock North")
    def lock_north(self) -> bool:
        return self._lock

    @lock_north.setter
    def lock_north(self, v: bool):
        self._lock = bool(v)
        self.changed.emit()

    @property
    @property_params(label="Show Full Path")
    def show_full_path(self) -> bool:
        return self._show_full_path

    @show_full_path.setter
    def show_full_path(self, v: bool):
        self._show_full_path = bool(v)
        self.changed.emit()
