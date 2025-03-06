import sys, os, glob, shutil, cv2, numpy as np, pprint
from scipy.optimize import minimize, least_squares
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import matplotlib

matplotlib.use("qt5agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib.patches as patches

# --- helper functions ---


def undistort_image(frame, mtx, dist):
    return cv2.undistort(frame, mtx, dist, None, mtx)


def isolate_laser(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([140, 70, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def triangulate_point(p, mtx, A, B, C, D):
    x, y = p
    u = (x - mtx[0, 2]) / mtx[0, 0]
    v = (y - mtx[1, 2]) / mtx[1, 1]
    denom = A * u + B * v + C
    if abs(denom) < 1e-6:
        return None
    t = -D / denom
    return np.array([t * u, t * v, t])


def fit_plane_and_project(A, B, C, D, points, mtx):
    normal = np.array([A, B, C], dtype=float)
    normal /= np.linalg.norm(normal)
    p0 = triangulate_point((0, 0), mtx, A, B, C, D)
    if np.allclose(normal, [0, 0, 1], atol=1e-6):
        u = np.array([1, 0, 0], dtype=float)
    else:
        u = np.cross(normal, [0, 0, 1])
        u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    projected = []
    for p in points:
        vec = p - p0
        projected.append([np.dot(vec, u), np.dot(vec, v)])
    return np.array(projected)


def circle_residuals(params, x, y):
    xc, yc, r = params
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r


def fit_circle(x, y):
    x_m, y_m = np.mean(x), np.mean(y)
    r0 = np.mean(np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2))
    res = least_squares(circle_residuals, [x_m, y_m, r0], args=(x, y))
    return res.x


# --- mplcanvas class for measure page ---
class mplcanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.lasso = LassoSelector(self.ax, onselect=self.onselect)
        self.points = []

        # store selected indices
        self.selected_indices = []
        self.circle = np.array([])

    def set_plot_limits(self, x_min, x_max, y_min, y_max):
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect("auto")

    def onselect(self, verts):
        path = Path(verts)
        ind = np.nonzero(path.contains_points(self.points))[0]
        # optionally highlight the selected points in red:
        self.ax.scatter(self.points[ind, 0], self.points[ind, 1], color="r")
        self.draw()
        self.circle = fit_circle(self.points[ind, 0], self.points[ind, 1])


# --- SceneCanvas for scene coordinate calibration ---
class SceneCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, onlasso_callback=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.onlasso_callback = onlasso_callback
        self.lasso = LassoSelector(self.ax, onselect=self.onselect)
        # holds raw 2d points (pixel coordinates)
        self.points = np.empty((0, 2))
        self.ax.set_xlim(0, 640)
        self.ax.set_ylim(0, 480)
        self.ax.set_aspect("equal", adjustable="box")

    def onselect(self, verts):
        path = Path(verts)
        ind = np.nonzero(path.contains_points(self.points))[0]
        selected_points = self.points[ind, :]
        if self.onlasso_callback:
            self.onlasso_callback(selected_points)
        # highlight selection
        self.ax.scatter(selected_points[:, 0], selected_points[:, 1], color="r")
        self.draw()


# --- main ui class ---
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("laser measurement tool")
        self.calib_images_dir = "calibration_images"
        if not os.path.exists(self.calib_images_dir):
            os.makedirs(self.calib_images_dir)
        self.max_capture_count = 20
        self.valid_capture_count = 0
        # load calibration data if available
        try:
            data = np.load("calibration_data.npz")
            self.mtx = data["mtx"]
            self.dist = data["dist"]
            self.is_calibrated = True
            print
        except Exception as e:
            print("calibration data not found, using defaults:", e)
            self.mtx = np.array(
                [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32
            )
            self.dist = np.zeros(5)
            self.is_calibrated = False
        # if calibration images exist, update capture count
        imgs = glob.glob(os.path.join(self.calib_images_dir, "*.png"))
        if imgs:
            self.valid_capture_count = min(len(imgs), self.max_capture_count)
        # default plane parameters for triangulation (intrinsics)
        #optimized plane: A=0.00022211785297654538, B=0.912921972529869, C=0.7072186832025387, D=-202.76804531284228
        #optimized plane: A=0.00022383650281094592, B=0.921240318305312, C=0.7278496055232874, D=-201.93702295740397
        #optimized plane: A=0.0002101289844084302, B=0.8116065433696171, C=0.8862311736685655, D=-212.28320500422865
        self.A = 0.0002101289844084302
        self.B = 0.8116065433696171
        self.C = 0.8862311736685655
        self.D = -212.28320500422865
        corners = fit_plane_and_project(
            self.A,
            self.B,
            self.C,
            self.D,
            [
                triangulate_point(c, self.mtx, self.A, self.B, self.C, self.D)
                for c in [(0, 0), (640, 0), (0, 480), (640, 480)]
            ],
            self.mtx,
        )
        self.plan_w = corners[1, 0]
        self.plan_h = corners[2, 1]
        # checkerboard defaults for intrinsics calibration
        self.checkerboard_size = (6, 9)
        self.square_size = 10.0
        # --- scene coordinate calibration variables ---
        self.scene_plane_guess = [self.A, self.B, self.C, self.D]
        self.scene_points_list = []  # list of np.arrays (selected points per cylinder)
        self.scene_known_radii = []  # corresponding list of known radii
        self.current_scene_selection = None
        self.init_ui()
        self.init_camera()
        self.current_points_3d = np.empty((0, 3))
        self.current_calib_frame = None
        self.timer = QtCore.QTimer()
        self.timer.start(30)
        # set initial page based on calibration status
        if self.is_calibrated:
            self.capture_count.setText(
                f"{self.max_capture_count} / {self.max_capture_count}"
            )
            self.capture_button.setText("already calibrated")
            self.capture_button.setEnabled(False)
            self.calib_message_label.setText("already calibrated")
            self.stacked_widget.setCurrentIndex(0)  # measure page
            self.timer.timeout.connect(self.update_frame)
        else:
            self.stacked_widget.setCurrentIndex(1)  # calibration page
            self.timer.timeout.connect(self.update_camera_calibration)

    def init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        nav_layout = QtWidgets.QHBoxLayout()
        self.measure_button = QtWidgets.QPushButton("measure")
        self.calibration_button = QtWidgets.QPushButton("calibration")
        self.measure_button.clicked.connect(lambda: self.switch_main_page(0))
        self.calibration_button.clicked.connect(lambda: self.switch_main_page(1))
        nav_layout.addWidget(self.measure_button)
        nav_layout.addWidget(self.calibration_button)
        main_layout.addLayout(nav_layout)
        self.stacked_widget = QtWidgets.QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        # --- measure page ---
        self.measure_page = QtWidgets.QWidget()
        measure_layout = QtWidgets.QHBoxLayout(self.measure_page)
        video_layout = QtWidgets.QVBoxLayout()
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640, 480)
        video_layout.addWidget(self.video_label)
        video_layout.addStretch()
        measure_layout.addLayout(video_layout)
        right_layout = QtWidgets.QVBoxLayout()
        self.plot_widget = mplcanvas(self, width=5, height=4, dpi=192)
        self.plot_widget.set_plot_limits(0, self.plan_w, 0, self.plan_h)
        right_layout.addWidget(self.plot_widget)
        self.nav_toolbar = NavigationToolbar(self.plot_widget, self)
        right_layout.addWidget(self.nav_toolbar)
        self.info_label = QtWidgets.QLabel("no points selected")
        self.info_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        right_layout.addWidget(self.info_label)
        measure_layout.addLayout(right_layout)
        self.stacked_widget.addWidget(self.measure_page)
        # --- calibration page ---
        self.calibration_page = QtWidgets.QWidget()
        calib_layout = QtWidgets.QVBoxLayout(self.calibration_page)
        calib_nav_layout = QtWidgets.QHBoxLayout()
        self.calib_intrinsics_button = QtWidgets.QPushButton("camera intrinsics")
        self.calib_scene_button = QtWidgets.QPushButton("scene coordinates")
        self.calib_intrinsics_button.clicked.connect(
            lambda: self.switch_calibration_subpage(0)
        )
        self.calib_scene_button.clicked.connect(
            lambda: self.switch_calibration_subpage(1)
        )
        calib_nav_layout.addWidget(self.calib_intrinsics_button)
        calib_nav_layout.addWidget(self.calib_scene_button)
        calib_layout.addLayout(calib_nav_layout)
        self.calibration_stacked = QtWidgets.QStackedWidget()
        calib_layout.addWidget(self.calibration_stacked)
        # --- intrinsics page (page 0) ---
        self.calib_intrinsics_page = QtWidgets.QWidget()
        intrinsics_layout = QtWidgets.QHBoxLayout(self.calib_intrinsics_page)
        video_layout_calib = QtWidgets.QVBoxLayout()
        self.video_label_camera_calibration = QtWidgets.QLabel()
        self.video_label_camera_calibration.setFixedSize(640, 480)
        video_layout_calib.addWidget(self.video_label_camera_calibration)
        video_layout_calib.addStretch()
        capture_layout = QtWidgets.QHBoxLayout()
        self.capture_button = QtWidgets.QPushButton("capture")
        self.capture_button.clicked.connect(self.on_image_calibration_capture)
        self.capture_count = QtWidgets.QLabel(
            f"{self.valid_capture_count} / {self.max_capture_count}"
        )
        capture_layout.addWidget(self.capture_button)
        capture_layout.addWidget(self.capture_count)
        video_layout_calib.addLayout(capture_layout)
        intrinsics_layout.addLayout(video_layout_calib)
        intrinsics_controls = QtWidgets.QVBoxLayout()
        self.dimension_label = QtWidgets.QLabel(
            "enter checkered pattern size (e.g. 6x9)"
        )
        self.dimension_input = QtWidgets.QLineEdit()
        self.dimension_input.setPlaceholderText("6x9")
        self.dimension_submit = QtWidgets.QPushButton("submit pattern")
        self.dimension_submit.clicked.connect(self.on_submit_dimension)
        self.square_size_label = QtWidgets.QLabel("enter square size in mm")
        self.square_size_input = QtWidgets.QLineEdit()
        self.square_size_input.setPlaceholderText("10")
        self.square_size_submit = QtWidgets.QPushButton("submit square size")
        self.square_size_submit.clicked.connect(self.on_submit_dimension)
        intrinsics_controls.addWidget(self.dimension_label)
        intrinsics_controls.addWidget(self.dimension_input)
        intrinsics_controls.addWidget(self.dimension_submit)
        intrinsics_controls.addWidget(self.square_size_label)
        intrinsics_controls.addWidget(self.square_size_input)
        intrinsics_controls.addWidget(self.square_size_submit)
        self.calib_message_label = QtWidgets.QLabel("")
        intrinsics_controls.addWidget(self.calib_message_label)
        self.reset_button = QtWidgets.QPushButton("reset calibration")
        self.reset_button.clicked.connect(self.on_reset_calibration)
        intrinsics_controls.addWidget(self.reset_button)
        intrinsics_controls.addStretch()
        intrinsics_layout.addLayout(intrinsics_controls)
        self.calibration_stacked.addWidget(self.calib_intrinsics_page)
        # --- scene coordinates page (page 1) ---
        self.scene_calib_page = QtWidgets.QWidget()
        scene_layout = QtWidgets.QHBoxLayout(self.scene_calib_page)
        video_layout_scene = QtWidgets.QVBoxLayout()
        self.video_label_scene = QtWidgets.QLabel()
        self.video_label_scene.setFixedSize(640, 480)
        video_layout_scene.addWidget(self.video_label_scene)
        video_layout_scene.addStretch()
        scene_layout.addLayout(video_layout_scene)
        scene_controls_layout = QtWidgets.QVBoxLayout()
        self.scene_canvas = SceneCanvas(
            self,
            width=5,
            height=4,
            dpi=192,
            onlasso_callback=self.on_scene_canvas_select,
        )
        scene_controls_layout.addWidget(self.scene_canvas)
        coeffs_layout = QtWidgets.QHBoxLayout()
        self.plane_A_input = QtWidgets.QLineEdit(str(self.scene_plane_guess[0]))
        self.plane_B_input = QtWidgets.QLineEdit(str(self.scene_plane_guess[1]))
        self.plane_C_input = QtWidgets.QLineEdit(str(self.scene_plane_guess[2]))
        self.plane_D_input = QtWidgets.QLineEdit(str(self.scene_plane_guess[3]))
        coeffs_layout.addWidget(QtWidgets.QLabel("A:"))
        coeffs_layout.addWidget(self.plane_A_input)
        coeffs_layout.addWidget(QtWidgets.QLabel("B:"))
        coeffs_layout.addWidget(self.plane_B_input)
        coeffs_layout.addWidget(QtWidgets.QLabel("C:"))
        coeffs_layout.addWidget(self.plane_C_input)
        coeffs_layout.addWidget(QtWidgets.QLabel("D:"))
        coeffs_layout.addWidget(self.plane_D_input)
        scene_controls_layout.addLayout(coeffs_layout)
        self.submit_plane_button = QtWidgets.QPushButton("submit plane guess")
        self.submit_plane_button.clicked.connect(self.on_submit_plane_guess)
        scene_controls_layout.addWidget(self.submit_plane_button)
        radius_layout = QtWidgets.QHBoxLayout()
        self.cylinder_radius_input = QtWidgets.QLineEdit()
        self.cylinder_radius_input.setPlaceholderText("enter cylinder radius (mm)")
        self.store_selection_button = QtWidgets.QPushButton("store selection")
        self.store_selection_button.clicked.connect(self.on_store_scene_selection)
        radius_layout.addWidget(self.cylinder_radius_input)
        radius_layout.addWidget(self.store_selection_button)
        scene_controls_layout.addLayout(radius_layout)
        self.optimize_button = QtWidgets.QPushButton("optimize")
        self.optimize_button.clicked.connect(self.on_optimize)
        scene_controls_layout.addWidget(self.optimize_button)
        self.scene_message_label = QtWidgets.QLabel("")
        scene_controls_layout.addWidget(self.scene_message_label)
        scene_controls_layout.addStretch()
        scene_layout.addLayout(scene_controls_layout)
        self.calibration_stacked.addWidget(self.scene_calib_page)
        self.stacked_widget.addWidget(self.calibration_page)

    def switch_main_page(self, index):
        self.stacked_widget.setCurrentIndex(index)
        self.timer.stop()
        if index == 0:
            self.timer.timeout.disconnect()
            self.timer.timeout.connect(self.update_frame)
        elif index == 1:
            self.timer.timeout.disconnect()
            self.timer.timeout.connect(self.update_camera_calibration)
        self.timer.start(30)

    def switch_calibration_subpage(self, index):
        self.calibration_stacked.setCurrentIndex(index)
        self.timer.stop()
        if index == 0:
            self.timer.timeout.disconnect()
            self.timer.timeout.connect(self.update_camera_calibration)
        elif index == 1:
            self.timer.timeout.disconnect()
            self.timer.timeout.connect(self.update_scene_calibration)
        self.timer.start(30)

    def update_camera_calibration(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_calib_frame = frame.copy()
        undistorted = undistort_image(frame, self.mtx, self.dist)
        vis_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
        h, w, ch = vis_rgb.shape
        qt_image = QtGui.QImage(vis_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.video_label_camera_calibration.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def update_scene_calibration(self):
        if not self.is_calibrated:
            self.scene_message_label.setText(
                "camera not calibrated – cannot perform scene calibration"
            )
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        undistorted = undistort_image(frame, self.mtx, self.dist)
        vis_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
        h, w, ch = vis_rgb.shape
        qt_image = QtGui.QImage(vis_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.video_label_scene.setPixmap(QtGui.QPixmap.fromImage(qt_image))
        mask = isolate_laser(undistorted)
        pts = cv2.findNonZero(mask)
        pts = pts.squeeze() if pts is not None else np.empty((0, 2))
        if pts.size:
            pts_flipped = pts.copy()
            pts_flipped[:, 1] = 480 - pts[:, 1]
        else:
            pts_flipped = pts
        self.scene_canvas.points = pts_flipped
        self.scene_canvas.ax.clear()
        self.scene_canvas.ax.scatter(
            pts_flipped[:, 0], pts_flipped[:, 1], s=2, color="b"
        )
        self.scene_canvas.ax.set_xlim(0, 640)
        self.scene_canvas.ax.set_ylim(0, 480)
        self.scene_canvas.draw()

    def on_image_calibration_capture(self):
        if self.current_calib_frame is None:
            self.calib_message_label.setText("no frame available")
            return
        gray = cv2.cvtColor(self.current_calib_frame, cv2.COLOR_BGR2GRAY)
        pattern_size = self.checkerboard_size
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            filename = os.path.join(
                self.calib_images_dir, f"img_{self.valid_capture_count:02d}.png"
            )
            cv2.imwrite(filename, self.current_calib_frame)
            self.valid_capture_count += 1
            self.capture_count.setText(
                f"{min(self.valid_capture_count, self.max_capture_count)} / {self.max_capture_count}"
            )
            self.calib_message_label.setText("image captured and valid")
            if self.valid_capture_count >= self.max_capture_count:
                self.capture_button.setText("calibrate")
                self.capture_button.clicked.disconnect()
                self.capture_button.clicked.connect(self.on_calibrate)
        else:
            self.calib_message_label.setText(
                "chessboard pattern not detected, image rejected"
            )

    def on_calibrate(self):
        objpoints = []
        imgpoints = []
        pattern_size = self.checkerboard_size
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(
            -1, 2
        )
        objp *= self.square_size
        images = glob.glob(os.path.join(self.calib_images_dir, "*.png"))
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                print(f"skipping: could not read file {fname}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners)
            else:
                print(f"no full pattern detected in {fname}, skipping.")
        if not objpoints:
            self.calib_message_label.setText(
                "no valid images found. calibration aborted."
            )
            return
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        if ret:
            np.savez("calibration_data.npz", mtx=mtx, dist=dist)
            self.calib_message_label.setText("calibration successful!")
            self.capture_button.setEnabled(False)
            self.is_calibrated = True
        else:
            self.calib_message_label.setText("calibration failed.")

    def on_submit_dimension(self):
        pattern_text = self.dimension_input.text().strip()
        if "x" in pattern_text:
            try:
                parts = pattern_text.lower().split("x")
                rows = int(parts[0])
                cols = int(parts[1])
                self.checkerboard_size = (rows, cols)
                self.calib_message_label.setText(
                    f"checkerboard size set to {rows}x{cols}"
                )
            except Exception as e:
                self.calib_message_label.setText("invalid checkerboard format")
        else:
            self.calib_message_label.setText("invalid checkerboard format")
        square_text = self.square_size_input.text().strip()
        try:
            self.square_size = float(square_text)
            current = self.calib_message_label.text()
            self.calib_message_label.setText(
                current + f" and square size set to {self.square_size} mm"
            )
        except Exception as e:
            self.calib_message_label.setText("invalid square size")

    def on_reset_calibration(self):
        if os.path.exists(self.calib_images_dir):
            shutil.rmtree(self.calib_images_dir)
            os.makedirs(self.calib_images_dir)
        if os.path.exists("calibration_data.npz"):
            os.remove("calibration_data.npz")
        self.valid_capture_count = 0
        self.capture_count.setText(f"0 / {self.max_capture_count}")
        self.capture_button.setText("capture")
        try:
            self.capture_button.clicked.disconnect()
        except Exception:
            pass
        self.capture_button.clicked.connect(self.on_image_calibration_capture)
        self.capture_button.setEnabled(True)
        self.calib_message_label.setText("calibration reset.")
        self.is_calibrated = False

    def on_scene_canvas_select(self, selected_points):
        self.current_scene_selection = selected_points
        self.scene_message_label.setText(
            f"{len(selected_points)} points selected via lasso"
        )

    def on_store_scene_selection(self):
        if (
            self.current_scene_selection is None
            or self.current_scene_selection.size == 0
        ):
            self.scene_message_label.setText("no points selected to store")
            return
        try:
            radius = float(self.cylinder_radius_input.text().strip())
        except Exception as e:
            self.scene_message_label.setText("invalid cylinder radius")
            return
        self.scene_points_list.append(self.current_scene_selection.copy())
        self.scene_known_radii.append(radius)
        self.scene_message_label.setText(
            f"stored selection #{len(self.scene_points_list)}; radius = {radius} mm"
        )
        self.current_scene_selection = None

    def on_submit_plane_guess(self):
        try:
            A_val = float(self.plane_A_input.text().strip())
            B_val = float(self.plane_B_input.text().strip())
            C_val = float(self.plane_C_input.text().strip())
            D_val = float(self.plane_D_input.text().strip())
            self.scene_plane_guess = [A_val, B_val, C_val, D_val]
            self.scene_message_label.setText(
                f"plane guess updated to: {self.scene_plane_guess}"
            )
        except Exception as e:
            self.scene_message_label.setText("invalid plane coefficient input")

    def on_optimize(self):
        if not self.scene_points_list or not self.scene_known_radii:
            self.scene_message_label.setText(
                "no cylinder data collected for optimization"
            )
            return

        def find_radius(params, points):
            A_opt, B_opt, C_opt, D_opt = params
            tri_pts = [
                triangulate_point(p, self.mtx, A_opt, B_opt, C_opt, D_opt)
                for p in points
            ]
            tri_pts = np.array([pt for pt in tri_pts if pt is not None])
            if tri_pts.size == 0:
                return None
            pts2d = fit_plane_and_project(A_opt, B_opt, C_opt, D_opt, tri_pts, self.mtx)
            if pts2d.size == 0:
                return None
            x, y = pts2d[:, 0], pts2d[:, 1]
            _, _, r = fit_circle(x, y)
            return r

        def objective(params):
            total_error = 0
            for known_radius, pts in zip(
                self.scene_known_radii, self.scene_points_list
            ):
                comp_r = find_radius(params, pts)
                if comp_r is None:
                    continue
                total_error += abs(comp_r - known_radius)
            return total_error

        res = minimize(objective, self.scene_plane_guess, method="Nelder-Mead")
        if res.success:
            A_opt, B_opt, C_opt, D_opt = res.x
            self.scene_message_label.setText(
                f"optimized plane: A={A_opt}, B={B_opt}, C={C_opt}, D={D_opt}"
            )
            self.A = A_opt
            self.B = B_opt
            self.C = C_opt
            self.D = D_opt
        else:
            self.scene_message_label.setText("optimization failed")

    def init_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print("failed to open camera")
            sys.exit(1)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        undistorted = undistort_image(frame, self.mtx, self.dist)
        mask = isolate_laser(undistorted)
        pts = cv2.findNonZero(mask)
        pts = pts.squeeze() if pts is not None else None
        if pts is not None:
            pts_flipped = pts.copy()
            pts_flipped[:, 1] = 480 - pts[:, 1]
        pts_3d = []
        if pts is not None:
            for p in pts_flipped:
                p3d = triangulate_point(p, self.mtx, self.A, self.B, self.C, self.D)
                if p3d is not None:
                    pts_3d.append(p3d)
        self.current_points_3d = np.array(pts_3d) if pts_3d else np.empty((0, 3))
        vis = undistorted.copy()
        if pts is not None:
            for p in pts:
                cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, ch = vis_rgb.shape
        qt_image = QtGui.QImage(vis_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
        xlim = self.plot_widget.ax.get_xlim()
        ylim = self.plot_widget.ax.get_ylim()
        self.plot_widget.ax.clear()
        if self.current_points_3d.shape[0] > 0:
            try:
                projected = fit_plane_and_project(
                    self.A, self.B, self.C, self.D, self.current_points_3d, self.mtx
                )
                self.plot_widget.points = projected
                self.plot_widget.ax.scatter(
                    projected[:, 0], projected[:, 1], color="r", s=1
                )
                self.plot_widget.ax.set_xlim(xlim)
                self.plot_widget.ax.set_ylim(ylim)
                if self.plot_widget.circle.size > 0:
                    circle = patches.Circle(
                        (self.plot_widget.circle[0], self.plot_widget.circle[1]),
                        self.plot_widget.circle[2],
                        edgecolor="r",
                        facecolor="none",
                        linewidth=2,
                    )
                    self.plot_widget.ax.add_patch(circle)
                    self.info_label.setText(f"radius: {self.plot_widget.circle[2]} mm")
                self.plot_widget.draw()
            except Exception as e:
                print("projection error:", e)
        if not self.is_calibrated:
            self.info_label.setText("you need to calibrate")

    def fit_circle_on_points(self):
        if self.current_points_3d.shape[0] == 0:
            self.info_label.setText("no points to fit")
            return
        try:
            projected = fit_plane_and_project(
                self.A, self.B, self.C, self.D, self.current_points_3d, self.mtx
            )
            x = projected[:, 0]
            y = projected[:, 1]
            xc, yc, r = fit_circle(x, y)
            circle = pg.QtGui.QGraphicsEllipseItem(xc - r, yc - r, 2 * r, 2 * r)
            circle.setPen(pg.mkPen("g", width=2))
            self.plot_widget.addItem(circle)
            info = f"center: ({xc:.2f}, {yc:.2f}), radius: {r:.2f} mm"
            self.info_label.setText(info)
        except Exception as e:
            self.info_label.setText("circle fit error")
            print("circle fit error:", e)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
