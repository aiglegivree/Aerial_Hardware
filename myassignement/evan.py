import cv2
import numpy as np


class MyAssignment:
    """Discover five gates, replay the learned lap, then return home."""

    def __init__(self):
        self.cruise_altitude = 1.35
        self.gate_vertical_bias = -1.0 / 12.0
        self.home_setpoint = [1.0, 4.0, self.cruise_altitude, 0.0]

        self.arena_center = np.array([4.0, 4.0], dtype=float)
        self.orbit_radius = 4.0
        self.first_lap_cone_radius = 3.0
        self.first_lap_cone_angle_offset = np.deg2rad(2.0)
        self.first_lap_cone_tol = 0.35
        self.first_lap_cone_step = 0.35
        self.first_lap_cone_target = None

        self.state = "TAKEOFF"
        self.debug = False
        self.debug_last_print_time = -999.0
        self.height_aligned = False
        self.time_since_start = 0.0
        self.last_seen_yaw = None
        self.last_seen_time = 0.0

        self.scan_phase = 0.0
        self.scan_center_yaw = None
        self.scan_last_sine = 0.0
        self.scan_half_swings = 0
        self.max_scan_half_swings = 2
        self.rotate_left_accum = 0.0
        self.rotate_left_last_yaw = None
        self.orbit_angle = None
        self.search_sweep_translation = 0.00001

        self.target_switch_count = 0
        self.target_switch_limit = 15
        self.force_rightmost_target = False
        self.last_target_center_px = None

        self.pass_through_timer = 0.0
        self.pass_through_heading = 0.0
        self.pass_through_altitude = self.cruise_altitude
        self.pass_through_duration = 1.35
        self.pass_through_distance = 0.8
        self.first_lap_gate_target = None
        self.first_lap_gate_tol_xy = 0.15
        self.first_lap_gate_tol_z = 0.2
        self.first_lap_gate_xy_step = 1.0
        self.first_lap_gate_z_step = 0.3
        self.advance_timer = 0.0
        self.advance_duration = 1.1
        self.advance_distance = 0.45

        self.camera_half_fov = np.deg2rad(35.0)
        self.camera_fov = 1.5
        self.camera_offset_body = np.array([0.03, 0.0, 0.01], dtype=float)
        self.camera_to_body = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        )

        self.gate_real_height = 0.4
        self.gate_focal_px = 320.0
        self.pending_gate_samples = []
        self.gate_bearing_observations = []
        self.min_triangulation_observations = 6
        self.max_triangulation_observations = 40
        self.min_triangulation_baseline = 0.8
        self.max_sample_area_after_baseline = 0.1
        self.min_sample_origin_spacing = 0.08
        self.triangulation_trigger_area = 0.05
        self.last_triangulation_failure_reason = None
        self.gate_center_backoff = 0.0

        self.side_sample_distance = 0.7
        self.side_sample_setpoint_distance = 0.25
        self.side_sample_duration = 30.0
        self.side_sample_target = None
        self.side_sample_right_direction = None
        self.side_sample_left_direction = None
        self.side_sample_orbit_center = None
        self.side_sample_orbit_radius = None
        self.side_sample_phase_origin = None
        self.side_sample_max_phase_distance = 0.75
        self.side_sample_altitude = self.cruise_altitude
        self.side_sample_timer = 0.0
        self.side_sample_used = False
        self.side_sample_phase = None
        self.side_sample_seen_left = False
        self.side_sample_seen_right = False
        self.side_sample_best_area = 0.0
        self.side_sample_best_position = None
        self.side_sample_phase_best_area = 0.0
        self.side_sample_last_center_px = None
        self.side_sample_near_zero_frames = 0
        self.side_sample_required_zero_frames = 5
        self.side_sample_low_area_frames = 0
        self.side_sample_required_low_area_frames = 2
        self.side_sample_switch_area_ratio = 0.5
        self.side_sample_area_switch_armed = False
        self.side_sample_ignore_visual_frames = 0
        self.side_sample_switch_ignore_frames = 30
        self.require_side_sample_before_store = True

        self.reswipe_backoff_distance = 0.45
        self.reswipe_setpoint_distance = 0.25
        self.reswipe_attempts = 0
        self.max_reswipe_attempts = 2
        self.reswipe_target = None
        self.reswipe_altitude = self.cruise_altitude

        self.learned_gates = []
        self.expected_gate_count = 5
        self.replay_lap = 0
        self.total_replay_laps = 2
        self.replay_trajectory = []
        self.replay_trajectory_types = []
        self.replay_trajectory_dirs = []
        self.replay_traj_index = 0
        self.last_printed_replay_type = None
        self.last_printed_replay_index = None
        self.last_replay_gate0_debug_time = -999.0
        self.replay_gate_crossing_armed_key = None
        self.replay_debug = False
        self.replay_traj_tol = 0.6
        self.replay_approach_tol = 0.25
        self.replay_gate_tol = 0.12
        self.replay_gate_z_tol = 0.12
        self.replay_gate_cross_margin = 0.16
        self.replay_gate_cross_command = 0.55
        self.replay_gate_align_hold_distance = 0.16
        self.replay_final_gate_cross_margin = 0.35
        self.replay_final_gate_cross_command = 0.7
        self.replay_traj_z_tol = 0.45
        self.replay_lookahead_distance = 2
        self.replay_switch_distance = 0.2
        self.replay_segment_lookahead = 2
        self.replay_xy_step = 2.6
        self.replay_z_step = 1.3
        self.replay_approach_distance = 0.25
        self.replay_min_approach_distance = 0.12
        self.replay_exit_distance = 0.6

    def debug_print(self, message, min_interval=0.0):
        if not self.debug:
            return
        if min_interval > 0.0 and self.time_since_start - self.debug_last_print_time < min_interval:
            return
        self.debug_last_print_time = self.time_since_start
        print(f"[assignment debug {self.time_since_start:.2f}s] {message}", flush=True)

    def current_xy(self, sensor_data):
        return np.array([sensor_data["x_global"], sensor_data["y_global"]], dtype=float)

    def current_xyz(self, sensor_data):
        return np.array(
            [sensor_data["x_global"], sensor_data["y_global"], sensor_data["z_global"]],
            dtype=float,
        )

    def hold_position(self, sensor_data, altitude=None, yaw=None):
        if altitude is None:
            altitude = sensor_data["z_global"]
        if yaw is None:
            yaw = sensor_data["yaw"]
        return [
            sensor_data["x_global"],
            sensor_data["y_global"],
            altitude,
            yaw,
        ]

    def forward_command(self, sensor_data, distance, altitude, yaw_target=None, heading=None):
        if heading is None:
            heading = sensor_data["yaw"]
        if yaw_target is None:
            yaw_target = heading
        return [
            sensor_data["x_global"] + distance * np.cos(heading),
            sensor_data["y_global"] + distance * np.sin(heading),
            altitude,
            yaw_target,
        ]

    @staticmethod
    def step_toward(current_xy, target_xy, max_step):
        delta = target_xy - current_xy
        distance = np.linalg.norm(delta)
        if distance <= 1e-6:
            return current_xy.copy()
        return current_xy + min(distance, max_step) * delta / distance

    def mark_gate_seen(self, sensor_data):
        self.last_seen_yaw = sensor_data["yaw"]
        self.last_seen_time = self.time_since_start
        self.scan_center_yaw = sensor_data["yaw"]

    def gate_altitude_target(self, altitude, error):
        z_error = error["dy"] - self.gate_vertical_bias
        if z_error > 0.0:
            z_step = np.clip(-0.75 * z_error, -0.16, 0.0)
        else:
            z_step = np.clip(-0.3 * z_error, 0.0, 0.07)
        return altitude + z_step, z_error

    def detect_gate(self, camera_data, track_switches=True):
        mask = self.compute_mask(camera_data)
        candidates = self.rectangle_candidates(mask, camera_data.shape)
        return self.choose_gate_candidate(candidates, track_switches=track_switches)

    def start_pass_through(self, sensor_data, altitude):
        self.state = "PASS_THROUGH"
        self.pass_through_timer = self.pass_through_duration
        self.pass_through_heading = sensor_data["yaw"]
        self.pass_through_altitude = altitude
        return self.pass_through_command(sensor_data, 0.0)

    def start_stored_gate_crossing(self, sensor_data):
        if self.first_lap_gate_target is None:
            self.debug_print("stored gate crossing requested without target, falling back to pass-through")
            return self.start_pass_through(sensor_data, sensor_data["z_global"])

        self.state = "GO_TO_STORED_GATE"
        gate = self.first_lap_gate_target["gate"]
        self.debug_print(
            f"starting stored gate crossing target=({gate[0]:.2f}, {gate[1]:.2f}, {gate[2]:.2f})"
        )
        return self.go_to_stored_gate_command(sensor_data)

    def handle_detected_gate(self, sensor_data, error):
        self.mark_gate_seen(sensor_data)
        self.state = "TRACK"
        self.reset_search_pattern(sensor_data["yaw"])
        return self.align_and_approach_command(sensor_data, error)

    def reset_search_pattern(self, current_yaw=None):
        self.scan_phase = 0.0
        self.scan_last_sine = 0.0
        self.scan_half_swings = 0
        self.rotate_left_accum = 0.0
        self.rotate_left_last_yaw = None
        if current_yaw is not None:
            self.scan_center_yaw = current_yaw

    @staticmethod
    def wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def current_angle_radius_target(self, sensor_data, altitude=None):
        relative = self.current_xy(sensor_data) - self.arena_center
        if np.linalg.norm(relative) < 1e-6:
            relative = np.array([0.0, -1.0], dtype=float)

        angle = np.arctan2(relative[1], relative[0]) + self.first_lap_cone_angle_offset
        target_xy = self.arena_center + self.first_lap_cone_radius * np.array(
            [np.cos(angle), np.sin(angle)],
            dtype=float,
        )
        yaw_target = np.arctan2(
            self.arena_center[1] - target_xy[1],
            self.arena_center[0] - target_xy[0],
        )
        if altitude is None:
            altitude = self.cruise_altitude
        return np.array([target_xy[0], target_xy[1], altitude, yaw_target], dtype=float)

    def compute_mask(self, camera_data):
        bgr = cv2.cvtColor(camera_data, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        lower = np.array([120, 40, 150], dtype=np.uint8)
        upper = np.array([160, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def rectangle_candidates(self, mask, img_shape):
        height, width = img_shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        image_area = float(height * width)
        candidates = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)

            center_x = float(box[:, 0].mean())
            center_y = float(box[:, 1].mean())
            candidates.append(
                {
                    "dx": float((center_x - width / 2.0) / (width / 2.0)),
                    "dy": float((center_y - height / 2.0) / (height / 2.0)),
                    "area_px": float(cv2.contourArea(contour)),
                    "area_rel": float(cv2.contourArea(contour) / image_area),
                    "pixel_height": float(np.max(box[:, 1]) - np.min(box[:, 1])),
                    "center_px": (center_x, center_y),
                    "image_shape": (height, width),
                }
            )

        return sorted(candidates, key=lambda item: item["area_rel"], reverse=True)

    def reset_target_switch_tracking(self):
        self.target_switch_count = 0
        self.force_rightmost_target = False
        self.last_target_center_px = None

    def update_target_switch_tracking(self, chosen):
        if len(self.learned_gates) >= self.expected_gate_count or chosen is None:
            return
        if self.force_rightmost_target:
            self.last_target_center_px = chosen["center_px"]
            return

        current_center = np.array(chosen["center_px"], dtype=float)
        if self.last_target_center_px is not None:
            previous_center = np.array(self.last_target_center_px, dtype=float)
            height, width = chosen["image_shape"]
            switch_threshold = 0.22 * np.linalg.norm([width, height])
            if np.linalg.norm(current_center - previous_center) > switch_threshold:
                self.target_switch_count += 1
                if self.target_switch_count > self.target_switch_limit:
                    self.force_rightmost_target = True

        self.last_target_center_px = chosen["center_px"]

    def choose_gate_candidate(self, candidates, track_switches=True):
        if not candidates:
            return None

        rightmost = max(candidates, key=lambda item: item["center_px"][0])
        if self.force_rightmost_target:
            if track_switches:
                self.update_target_switch_tracking(rightmost)
            return rightmost

        largest = max(candidates, key=lambda item: item["area_rel"])
        chosen = largest if largest["area_rel"] >= 1.3 * rightmost["area_rel"] else rightmost

        if track_switches:
            self.update_target_switch_tracking(chosen)
            if self.force_rightmost_target:
                self.last_target_center_px = rightmost["center_px"]
                return rightmost
        return chosen

    def gate_world_estimate(self, sensor_data, error):
        pixel_height = max(error.get("pixel_height", 0.0), 1.0)
        gate_distance = self.gate_focal_px * self.gate_real_height / pixel_height
        gate_distance = float(np.clip(gate_distance, 0.5, 4.0))
        bearing = sensor_data["yaw"] - error["dx"] * self.camera_half_fov
        gate_x = sensor_data["x_global"] + gate_distance * np.cos(bearing)
        gate_y = sensor_data["y_global"] + gate_distance * np.sin(bearing)
        gate_z = sensor_data["z_global"]
        return [float(gate_x), float(gate_y), float(gate_z)]

    def body_to_world_rotation(self, sensor_data):
        roll = sensor_data["roll"]
        pitch = sensor_data["pitch"]
        yaw = sensor_data["yaw"]
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        r_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cr, -sr],
                [0.0, sr, cr],
            ],
            dtype=float,
        )
        r_y = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=float,
        )
        r_z = np.array(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return r_z @ r_y @ r_x

    def camera_ray_world(self, sensor_data, error):
        height, width = error["image_shape"]
        center_x, center_y = error["center_px"]
        focal_px = width / (2.0 * np.tan(self.camera_fov / 2.0))
        pixel_x = center_x - width / 2.0
        pixel_y = center_y - height / 2.0

        ray_camera = np.array([pixel_x, pixel_y, focal_px], dtype=float)
        ray_body = self.camera_to_body @ ray_camera
        rotation = self.body_to_world_rotation(sensor_data)
        ray_world = rotation @ ray_body
        ray_norm = np.linalg.norm(ray_world)
        if ray_norm < 1e-6:
            return None

        camera_origin = self.current_xyz(sensor_data) + rotation @ self.camera_offset_body
        return camera_origin, ray_world / ray_norm

    def collect_gate_sample(self, sensor_data, error):
        if len(self.learned_gates) >= self.expected_gate_count:
            return

        self.pending_gate_samples.append(self.gate_world_estimate(sensor_data, error))
        if len(self.pending_gate_samples) > 12:
            self.pending_gate_samples = self.pending_gate_samples[-12:]

        ray = self.camera_ray_world(sensor_data, error)
        if ray is None:
            return

        camera_origin, ray_direction = ray
        baseline_ready = self.triangulation_baseline() >= self.min_triangulation_baseline
        if baseline_ready and error["area_rel"] > self.max_sample_area_after_baseline:
            return

        if self.gate_bearing_observations:
            nearest_origin_distance = min(
                np.linalg.norm(camera_origin - obs["origin"])
                for obs in self.gate_bearing_observations
            )
            if nearest_origin_distance < self.min_sample_origin_spacing:
                return

        centered_score = max(0.05, 1.0 - abs(error["dx"]))
        area_score = max(error["area_rel"], 1e-4)
        self.gate_bearing_observations.append(
            {
                "origin": camera_origin,
                "direction": ray_direction,
                "weight": float(area_score * centered_score * centered_score),
            }
        )
        if len(self.gate_bearing_observations) > self.max_triangulation_observations:
            self.gate_bearing_observations = self.select_best_bearing_observations()

    def select_best_bearing_observations(self):
        if len(self.gate_bearing_observations) <= self.max_triangulation_observations:
            return self.gate_bearing_observations

        remaining = list(self.gate_bearing_observations)
        first_index = max(range(len(remaining)), key=lambda idx: remaining[idx]["weight"])
        selected = [remaining.pop(first_index)]

        while remaining and len(selected) < self.max_triangulation_observations:
            best_index = 0
            best_score = -np.inf
            for index, observation in enumerate(remaining):
                min_distance = min(
                    np.linalg.norm(observation["origin"] - kept["origin"])
                    for kept in selected
                )
                score = observation["weight"] * (0.35 + min_distance)
                if score > best_score:
                    best_score = score
                    best_index = index
            selected.append(remaining.pop(best_index))

        return selected

    def reset_gate_localization(self):
        self.pending_gate_samples = []
        self.gate_bearing_observations = []
        self.side_sample_target = None
        self.side_sample_right_direction = None
        self.side_sample_left_direction = None
        self.side_sample_orbit_center = None
        self.side_sample_orbit_radius = None
        self.side_sample_phase_origin = None
        self.side_sample_timer = 0.0
        self.side_sample_used = False
        self.side_sample_phase = None
        self.side_sample_seen_left = False
        self.side_sample_seen_right = False
        self.side_sample_best_area = 0.0
        self.side_sample_best_position = None
        self.side_sample_phase_best_area = 0.0
        self.side_sample_last_center_px = None
        self.side_sample_near_zero_frames = 0
        self.side_sample_low_area_frames = 0
        self.side_sample_area_switch_armed = False
        self.side_sample_ignore_visual_frames = 0
        self.reswipe_attempts = 0
        self.reswipe_target = None
        self.reswipe_altitude = self.cruise_altitude

    def begin_reswipe_backoff(self, sensor_data, altitude):
        backward_direction = -np.array(
            [np.cos(sensor_data["yaw"]), np.sin(sensor_data["yaw"])],
            dtype=float,
        )
        self.reswipe_target = self.current_xy(sensor_data) + self.reswipe_backoff_distance * backward_direction
        self.reswipe_altitude = altitude
        self.reswipe_attempts += 1
        self.state = "RESWIPE_BACKOFF"
        return self.hold_position(sensor_data, altitude=altitude, yaw=sensor_data["yaw"])

    def post_side_sample_resolution(self, sensor_data, error, altitude):
        z_target, _ = self.gate_altitude_target(altitude, error)
        self.debug_print(
            f"post side sample resolution area={error['area_rel']:.4f}, "
            f"observations={len(self.gate_bearing_observations)}, "
            f"baseline={self.triangulation_baseline():.3f}"
        )

        if self.maybe_store_gate(sensor_data, error):
            self.debug_print("post side sample stored triangulated gate")
            return self.start_stored_gate_crossing(sensor_data)

        if self.reswipe_attempts < self.max_reswipe_attempts:
            self.debug_print(
                f"post side sample triangulation failed ({self.last_triangulation_failure_reason}), reswipe"
            )
            return self.begin_reswipe_backoff(sensor_data, altitude)

        if self.accept_best_gate_estimate(sensor_data, error):
            self.debug_print("post side sample accepted best gate estimate")
            return self.start_stored_gate_crossing(sensor_data)

        self.debug_print(
            f"post side sample unresolved, holding ({self.last_triangulation_failure_reason})"
        )
        return self.hold_position(sensor_data, altitude=z_target, yaw=sensor_data["yaw"])

    def triangulation_baseline(self):
        if len(self.gate_bearing_observations) < 2:
            return 0.0
        origins = np.array([obs["origin"] for obs in self.gate_bearing_observations], dtype=float)
        separations = origins[:, None, :] - origins[None, :, :]
        return float(np.max(np.linalg.norm(separations, axis=2)))

    def triangulate_gate_position(self):
        if len(self.gate_bearing_observations) < self.min_triangulation_observations:
            self.last_triangulation_failure_reason = "not enough samples"
            return None

        baseline = self.triangulation_baseline()
        if baseline < self.min_triangulation_baseline:
            self.last_triangulation_failure_reason = (
                f"baseline too small ({baseline:.3f} < {self.min_triangulation_baseline:.3f})"
            )
            return None

        a_matrix = np.zeros((3, 3), dtype=float)
        b_vector = np.zeros(3, dtype=float)
        identity = np.eye(3, dtype=float)
        weight_sum = 0.0

        for observation in self.gate_bearing_observations:
            direction = observation["direction"]
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue

            direction = direction / norm
            normal_matrix = identity - np.outer(direction, direction)
            weight = observation["weight"]
            a_matrix += weight * normal_matrix
            b_vector += weight * normal_matrix @ observation["origin"]
            weight_sum += weight

        if weight_sum <= 1e-6:
            self.last_triangulation_failure_reason = "zero weight sum"
            return None

        condition = np.linalg.cond(a_matrix)
        if condition > 40.0:
            self.last_triangulation_failure_reason = f"ill-conditioned solve ({float(condition):.2f})"
            return None

        gate_position = np.linalg.solve(a_matrix, b_vector)
        self.last_triangulation_failure_reason = None
        return [float(gate_position[0]), float(gate_position[1]), float(gate_position[2])]

    def shift_gate_toward_drone(self, sensor_data, gate_position):
        gate = np.array(gate_position, dtype=float)
        direction = gate[:2] - self.current_xy(sensor_data)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return gate_position

        gate[:2] -= self.gate_center_backoff * direction / norm
        return [float(gate[0]), float(gate[1]), float(gate[2])]

    def store_gate_candidate(self, sensor_data, gate_candidate, allow_close=False):
        gate_candidate = self.shift_gate_toward_drone(sensor_data, gate_candidate)

        if self.learned_gates and not allow_close:
            previous = np.array(self.learned_gates[-1]["gate"], dtype=float)
            if np.linalg.norm(np.array(gate_candidate, dtype=float) - previous) < 1.0:
                self.last_triangulation_failure_reason = "candidate too close to previous gate"
                self.reset_gate_localization()
                self.debug_print("store gate rejected: candidate too close to previous gate")
                return False

        gate_bundle = {
            "gate": gate_candidate,
            "heading": float(sensor_data["yaw"]),
        }
        self.learned_gates.append(gate_bundle)
        self.first_lap_gate_target = gate_bundle.copy()
        self.debug_print(
            f"stored gate #{len(self.learned_gates)} at "
            f"({gate_candidate[0]:.2f}, {gate_candidate[1]:.2f}, {gate_candidate[2]:.2f}), "
            f"heading={sensor_data['yaw']:.2f}, allow_close={allow_close}"
        )
        self.reset_gate_localization()
        return True

    def accept_best_gate_estimate(self, sensor_data, error):
        if len(self.learned_gates) >= self.expected_gate_count:
            self.reset_gate_localization()
            return False

        if not self.pending_gate_samples:
            self.collect_gate_sample(sensor_data, error)
        if not self.pending_gate_samples:
            return False

        gate_candidate = np.mean(np.array(self.pending_gate_samples, dtype=float), axis=0).tolist()
        return self.store_gate_candidate(
            sensor_data,
            gate_candidate,
            allow_close=True,
        )

    def maybe_store_gate(self, sensor_data, error):
        if len(self.learned_gates) >= self.expected_gate_count:
            self.reset_gate_localization()
            return False

        self.collect_gate_sample(sensor_data, error)
        gate_candidate = self.triangulate_gate_position()
        if gate_candidate is None:
            self.debug_print(
                f"triangulation failed: {self.last_triangulation_failure_reason}, "
                f"observations={len(self.gate_bearing_observations)}, "
                f"baseline={self.triangulation_baseline():.3f}",
                min_interval=0.5,
            )
            return False

        return self.store_gate_candidate(sensor_data, gate_candidate)

    def set_side_sample_phase(self, target, phase, reset_area=True, reason="", current_xy=None):
        self.side_sample_target = None if target is None else np.array(target, dtype=float).copy()
        self.side_sample_phase = phase
        self.side_sample_phase_origin = None if current_xy is None else np.array(current_xy, dtype=float).copy()
        if phase == "LEFT_SCAN":
            self.side_sample_seen_left = True
        if phase == "RIGHT_SCAN":
            self.side_sample_seen_right = True
        if reset_area:
            self.side_sample_phase_best_area = 0.0
        self.side_sample_last_center_px = None
        self.side_sample_near_zero_frames = 0
        self.side_sample_low_area_frames = 0
        self.side_sample_area_switch_armed = reset_area
        if phase in ("LEFT_SCAN", "RIGHT_SCAN") and not reset_area:
            self.side_sample_ignore_visual_frames = self.side_sample_switch_ignore_frames
        else:
            self.side_sample_ignore_visual_frames = 0
        reason_text = f" reason={reason}" if reason else ""
        target_text = (
            "none"
            if self.side_sample_target is None
            else f"({self.side_sample_target[0]:.2f}, {self.side_sample_target[1]:.2f})"
        )
        self.debug_print(
            f"side sample phase -> {phase}, target={target_text}, best_area={self.side_sample_best_area:.4f}, "
            f"phase_best_area={self.side_sample_phase_best_area:.4f}, "
            f"ignore_visual_frames={self.side_sample_ignore_visual_frames}{reason_text}"
        )

    def begin_side_sample(self, sensor_data, altitude, initial_error=None):
        self.side_sample_left_direction = np.array(
            [-np.sin(sensor_data["yaw"]), np.cos(sensor_data["yaw"])],
            dtype=float,
        )
        self.side_sample_right_direction = -self.side_sample_left_direction
        current_xy = self.current_xy(sensor_data)
        self.side_sample_orbit_center = None
        self.side_sample_orbit_radius = None
        if initial_error is not None:
            gate_estimate = np.array(self.gate_world_estimate(sensor_data, initial_error), dtype=float)
            self.side_sample_orbit_center = gate_estimate[:2]
            self.side_sample_orbit_radius = float(np.linalg.norm(current_xy - self.side_sample_orbit_center))

        self.side_sample_altitude = altitude
        self.side_sample_timer = self.side_sample_duration
        self.side_sample_used = True
        self.side_sample_best_area = initial_error["area_rel"] if initial_error is not None else 0.0
        self.side_sample_best_position = current_xy.copy()
        self.side_sample_last_center_px = initial_error["center_px"] if initial_error is not None else None
        self.side_sample_near_zero_frames = 0
        self.state = "SIDE_SAMPLE"
        self.debug_print(
            f"begin side sample altitude={altitude:.2f}, current=({current_xy[0]:.2f}, {current_xy[1]:.2f}), "
            f"left_dir=({self.side_sample_left_direction[0]:.2f}, {self.side_sample_left_direction[1]:.2f}), "
            f"right_dir=({self.side_sample_right_direction[0]:.2f}, {self.side_sample_right_direction[1]:.2f}), "
            f"orbit_center={None if self.side_sample_orbit_center is None else tuple(np.round(self.side_sample_orbit_center, 2))}, "
            f"orbit_radius={self.side_sample_orbit_radius}, initial_area={self.side_sample_best_area:.4f}"
        )
        self.set_side_sample_phase(None, "LEFT_SCAN", reason="start", current_xy=current_xy)
        return self.hold_position(sensor_data, altitude=altitude, yaw=sensor_data["yaw"])

    def choose_side_sample_candidate(self, candidates):
        if not candidates:
            return None
        if self.side_sample_last_center_px is None:
            return self.choose_gate_candidate(candidates, track_switches=False)

        previous = np.array(self.side_sample_last_center_px, dtype=float)
        closest = min(
            candidates,
            key=lambda item: np.linalg.norm(np.array(item["center_px"], dtype=float) - previous),
        )
        height, width = closest["image_shape"]
        jump = np.linalg.norm(np.array(closest["center_px"], dtype=float) - previous)
        max_jump = 0.30 * np.linalg.norm([width, height])
        if jump > max_jump and closest["area_rel"] < self.side_sample_best_area * 0.25:
            return None
        return closest

    def switch_side_sample_to_left(self, current_xy):
        if self.side_sample_left_direction is None or self.side_sample_phase != "RIGHT_SCAN":
            return
        self.set_side_sample_phase(None, "LEFT_SCAN", reset_area=False, reason="visual switch", current_xy=current_xy)

    def switch_side_sample_to_right(self, current_xy):
        if self.side_sample_right_direction is None or self.side_sample_phase != "LEFT_SCAN":
            return
        self.set_side_sample_phase(None, "RIGHT_SCAN", reset_area=False, reason="visual switch", current_xy=current_xy)

    def switch_side_sample_to_best(self, reason="", current_xy=None):
        if not (self.side_sample_seen_left and self.side_sample_seen_right):
            if self.side_sample_phase == "LEFT_SCAN" and current_xy is not None:
                self.debug_print("side sample needs both sides before return, forcing RIGHT_SCAN")
                self.switch_side_sample_to_right(current_xy)
            return
        if self.side_sample_best_position is None:
            return
        self.set_side_sample_phase(self.side_sample_best_position, "RETURN_BEST", reason=reason, current_xy=current_xy)

    def side_sample_arc_command_xy(self, current_xy):
        if self.side_sample_phase == "LEFT_SCAN":
            preferred_direction = self.side_sample_left_direction
        elif self.side_sample_phase == "RIGHT_SCAN":
            preferred_direction = self.side_sample_right_direction
        else:
            return self.step_toward(current_xy, self.side_sample_target, self.side_sample_setpoint_distance)

        if (
            self.side_sample_orbit_center is None
            or self.side_sample_orbit_radius is None
            or self.side_sample_orbit_radius < 0.2
            or preferred_direction is None
        ):
            return current_xy + self.side_sample_setpoint_distance * preferred_direction

        radial = current_xy - self.side_sample_orbit_center
        radial_norm = np.linalg.norm(radial)
        if radial_norm < 1e-6:
            return current_xy + self.side_sample_setpoint_distance * preferred_direction

        tangent = np.array([-radial[1], radial[0]], dtype=float) / radial_norm
        if np.dot(tangent, preferred_direction) < 0.0:
            tangent = -tangent

        trial_xy = current_xy + self.side_sample_setpoint_distance * tangent
        trial_radial = trial_xy - self.side_sample_orbit_center
        trial_norm = np.linalg.norm(trial_radial)
        if trial_norm < 1e-6:
            return trial_xy

        return self.side_sample_orbit_center + self.side_sample_orbit_radius * trial_radial / trial_norm

    def side_sample_command(self, sensor_data, camera_data, dt):
        self.side_sample_timer -= dt
        if (
            self.side_sample_phase == "RETURN_BEST"
            and self.side_sample_target is None
        ):
            self.debug_print("side sample target missing, returning to TRACK")
            self.state = "TRACK"
            return self.hold_position(sensor_data)

        mask = self.compute_mask(camera_data)
        candidates = self.rectangle_candidates(mask, camera_data.shape)
        error = self.choose_side_sample_candidate(candidates)
        usable_error = None
        yaw_target = sensor_data["yaw"]
        current_xy = self.current_xy(sensor_data)
        ignore_visual = (
            self.side_sample_phase in ("LEFT_SCAN", "RIGHT_SCAN")
            and self.side_sample_ignore_visual_frames > 0
        )
        if ignore_visual:
            self.side_sample_ignore_visual_frames -= 1

        if error is not None:
            self.mark_gate_seen(sensor_data)
            yaw_correction = -0.3 * error["dx"]
            yaw_target = sensor_data["yaw"] + np.clip(yaw_correction, -np.deg2rad(14), np.deg2rad(14))

            current_area = error["area_rel"]
            switch_area_threshold = self.side_sample_switch_area_ratio * self.side_sample_phase_best_area
            low_area = (
                self.side_sample_phase in ("LEFT_SCAN", "RIGHT_SCAN")
                and self.side_sample_phase_best_area > 0.0
                and current_area <= switch_area_threshold
            )
            if low_area:
                self.side_sample_low_area_frames += 1
            else:
                self.side_sample_low_area_frames = 0
                if self.side_sample_phase in ("LEFT_SCAN", "RIGHT_SCAN"):
                    self.side_sample_area_switch_armed = True

            should_switch_side = (
                not ignore_visual
                and self.side_sample_area_switch_armed
                and
                low_area
                and self.side_sample_low_area_frames >= self.side_sample_required_low_area_frames
            )

            if should_switch_side:
                self.debug_print(
                    f"side sample area switch phase={self.side_sample_phase}, "
                    f"area={current_area:.4f}, threshold={switch_area_threshold:.4f}, "
                    f"low_frames={self.side_sample_low_area_frames}"
                )
                if self.side_sample_phase == "LEFT_SCAN":
                    self.switch_side_sample_to_right(current_xy)
                else:
                    self.switch_side_sample_to_best(reason="area drop after right scan", current_xy=current_xy)
            else:
                self.side_sample_near_zero_frames = 0
                self.side_sample_last_center_px = error["center_px"]
                self.collect_gate_sample(sensor_data, error)
                usable_error = error
                if self.side_sample_orbit_center is None:
                    gate_estimate = np.array(self.gate_world_estimate(sensor_data, error), dtype=float)
                    self.side_sample_orbit_center = gate_estimate[:2]
                    self.side_sample_orbit_radius = float(np.linalg.norm(current_xy - self.side_sample_orbit_center))

                if current_area > self.side_sample_best_area:
                    self.side_sample_best_area = current_area
                    self.side_sample_best_position = current_xy.copy()
                if current_area > self.side_sample_phase_best_area:
                    self.side_sample_phase_best_area = current_area
        elif not ignore_visual:
            self.side_sample_near_zero_frames += 1
            self.debug_print(
                f"side sample no gate phase={self.side_sample_phase}, "
                f"zero_frames={self.side_sample_near_zero_frames}/{self.side_sample_required_zero_frames}"
            )
            if (
                self.side_sample_phase == "LEFT_SCAN"
                and self.side_sample_near_zero_frames >= self.side_sample_required_zero_frames
            ):
                self.debug_print("side sample no-frame switch LEFT_SCAN -> RIGHT_SCAN")
                self.switch_side_sample_to_right(current_xy)
            if (
                self.side_sample_phase == "RIGHT_SCAN"
                and self.side_sample_near_zero_frames >= self.side_sample_required_zero_frames
            ):
                self.debug_print("side sample no-frame switch RIGHT_SCAN -> RETURN_BEST")
                self.switch_side_sample_to_best(reason="no gate after right scan", current_xy=current_xy)

        reached_target = (
            self.side_sample_phase == "RETURN_BEST"
            and self.side_sample_target is not None
            and np.linalg.norm(self.side_sample_target - current_xy) < 0.05
        )

        if self.side_sample_timer <= 0.0 and self.side_sample_phase != "RETURN_BEST":
            self.debug_print(f"side sample timer expired in {self.side_sample_phase}, returning to best")
            self.switch_side_sample_to_best(reason="timer", current_xy=current_xy)
            self.side_sample_timer = 1.5
            reached_target = False

        phase_distance = 0.0
        if self.side_sample_phase_origin is not None and self.side_sample_phase in ("LEFT_SCAN", "RIGHT_SCAN"):
            phase_distance = float(np.linalg.norm(current_xy - self.side_sample_phase_origin))

        if (reached_target and self.side_sample_phase == "RETURN_BEST") or self.side_sample_timer <= 0.0:
            self.debug_print(
                f"side sample ending phase={self.side_sample_phase}, reached_best={reached_target}, "
                f"timer={self.side_sample_timer:.2f}, usable_error={usable_error is not None}"
            )
            self.side_sample_target = None
            self.side_sample_timer = 0.0
            self.side_sample_phase = None
            self.state = "TRACK"
            if usable_error is None:
                usable_error = self.detect_gate(camera_data, track_switches=False)
            if usable_error is not None:
                return self.post_side_sample_resolution(sensor_data, usable_error, self.side_sample_altitude)
            if self.reswipe_attempts < self.max_reswipe_attempts:
                self.debug_print("side sample ended without usable gate, beginning reswipe")
                return self.begin_reswipe_backoff(sensor_data, self.side_sample_altitude)
            self.debug_print("side sample ended without usable gate, holding position")
            return self.hold_position(sensor_data, altitude=self.side_sample_altitude, yaw=yaw_target)

        target_distance = (
            float(np.linalg.norm(self.side_sample_target - current_xy))
            if self.side_sample_target is not None
            else None
        )
        area_text = "none" if error is None else f"{error['area_rel']:.4f}"
        target_distance_text = "none" if target_distance is None else f"{target_distance:.2f}"
        ignore_text = (
            f", ignoring_switch=True, ignore_remaining={self.side_sample_ignore_visual_frames}"
            if ignore_visual
            else ""
        )
        self.debug_print(
            f"side sample status phase={self.side_sample_phase}, target_dist={target_distance_text}, "
            f"timer={self.side_sample_timer:.2f}, area={area_text}, "
            f"best_area={self.side_sample_best_area:.4f}, phase_best={self.side_sample_phase_best_area:.4f}, "
            f"low_area_frames={self.side_sample_low_area_frames}, area_armed={self.side_sample_area_switch_armed}, "
            f"phase_distance={phase_distance:.2f}, ignore_visual={self.side_sample_ignore_visual_frames}"
            f"{ignore_text}",
            min_interval=0.5,
        )

        command_xy = self.side_sample_arc_command_xy(current_xy)
        return [
            float(command_xy[0]),
            float(command_xy[1]),
            self.side_sample_altitude,
            yaw_target,
        ]

    def reswipe_backoff_command(self, sensor_data, camera_data):
        if self.reswipe_target is None:
            self.state = "TRACK"
            return self.hold_position(sensor_data)

        current_xy = self.current_xy(sensor_data)
        distance = np.linalg.norm(self.reswipe_target - current_xy)
        if distance > 0.05:
            command_xy = self.step_toward(current_xy, self.reswipe_target, self.reswipe_setpoint_distance)
            return [
                float(command_xy[0]),
                float(command_xy[1]),
                self.reswipe_altitude,
                sensor_data["yaw"],
            ]

        self.reswipe_target = None
        self.side_sample_used = False
        error = self.detect_gate(camera_data, track_switches=False)
        if error is not None:
            return self.begin_side_sample(sensor_data, self.reswipe_altitude, error)

        self.state = "TRACK"
        return self.hold_position(sensor_data, altitude=self.reswipe_altitude, yaw=sensor_data["yaw"])

    def build_replay_trajectory(self, entry_position=None):
        if len(self.learned_gates) < self.expected_gate_count:
            self.replay_trajectory = []
            self.replay_trajectory_types = []
            self.replay_trajectory_dirs = []
            self.replay_traj_index = 0
            return

        trajectory = []
        trajectory_types = []
        trajectory_dirs = []
        for index, gate_bundle in enumerate(self.learned_gates):
            gate_point = np.array(gate_bundle["gate"], dtype=float)
            heading = gate_bundle.get("heading")
            if heading is None:
                previous_gate = np.array(
                    self.learned_gates[(index - 1) % len(self.learned_gates)]["gate"],
                    dtype=float,
                )
                approach_direction = gate_point[:2] - previous_gate[:2]
                heading = np.arctan2(approach_direction[1], approach_direction[0])

            approach_direction = np.array([np.cos(heading), np.sin(heading)], dtype=float)
            approach_point = self.replay_approach_point(gate_point, approach_direction)
            exit_point = gate_point.copy()
            exit_point[:2] += self.replay_exit_distance * approach_direction

            direction_list = approach_direction.tolist()
            trajectory.extend([approach_point.tolist(), gate_point.tolist(), exit_point.tolist()])
            trajectory_types.extend(["approach", "gate", "exit"])
            trajectory_dirs.extend([direction_list, direction_list, direction_list])

        if entry_position is not None and trajectory:
            entry_point = np.array(entry_position, dtype=float)
            first_target = np.array(trajectory[0], dtype=float)
            if np.linalg.norm(first_target - entry_point) > self.replay_switch_distance:
                trajectory.insert(0, entry_point.tolist())
                trajectory_types.insert(0, "entry")
                trajectory_dirs.insert(0, trajectory_dirs[0])

        self.replay_lap = 0
        self.replay_trajectory = trajectory
        self.replay_trajectory_types = trajectory_types
        self.replay_trajectory_dirs = trajectory_dirs
        self.replay_traj_index = 0
        self.last_printed_replay_type = None
        self.last_printed_replay_index = None

        if self.replay_debug and len(trajectory) >= 3:
            offset = 1 if trajectory_types and trajectory_types[0] == "entry" else 0
            first_approach = trajectory[offset]
            first_gate = trajectory[offset + 1]
            first_exit = trajectory[offset + 2]
            first_direction = trajectory_dirs[offset + 1]
            print(
                "[assignment replay] gate0 path "
                f"approach=({first_approach[0]:.2f},{first_approach[1]:.2f},{first_approach[2]:.2f}) "
                f"gate=({first_gate[0]:.2f},{first_gate[1]:.2f},{first_gate[2]:.2f}) "
                f"exit=({first_exit[0]:.2f},{first_exit[1]:.2f},{first_exit[2]:.2f}) "
                f"dir=({first_direction[0]:.2f},{first_direction[1]:.2f})",
                flush=True,
            )

    def replay_current_tolerance(self):
        if not self.replay_trajectory_types:
            return self.replay_traj_tol
        if self.replay_trajectory_types[self.replay_traj_index] == "gate":
            return self.replay_gate_tol
        return self.replay_traj_tol

    def replay_gate_number(self, trajectory_index):
        if not self.replay_trajectory_types:
            return None
        offset = 1 if self.replay_trajectory_types[0] == "entry" else 0
        relative_index = trajectory_index - offset
        if relative_index < 0:
            return None
        return relative_index // 3

    def replay_target_key(self):
        return (self.replay_lap, self.replay_traj_index)

    def is_final_replay_gate_target(self):
        if not self.replay_trajectory_types:
            return False
        return (
            self.replay_lap == self.total_replay_laps - 1
            and self.replay_trajectory_types[self.replay_traj_index] == "gate"
            and self.replay_gate_number(self.replay_traj_index) == self.expected_gate_count - 1
        )

    def replay_segment_index(self, xy_position):
        relative = np.array(xy_position, dtype=float) - self.arena_center
        norm = np.linalg.norm(relative)
        if norm < 1e-6:
            return -1

        angle = np.arctan2(relative[1] / norm, relative[0] / norm) + np.pi
        segment_size = np.pi / (self.expected_gate_count + 1)
        for index in range(self.expected_gate_count + 1):
            lower = (2 * index - 0.5) * segment_size % (2 * np.pi)
            upper = (2 * index + 0.5) * segment_size % (2 * np.pi)
            if index == 0:
                if angle >= lower or angle <= upper:
                    return index
            elif lower <= angle <= upper:
                return index
        return -1

    def replay_approach_point(self, gate_point, approach_direction):
        gate_segment = self.replay_segment_index(gate_point[:2])
        approach_distance = self.replay_approach_distance
        while approach_distance >= self.replay_min_approach_distance:
            approach_point = gate_point.copy()
            approach_point[:2] -= approach_distance * approach_direction
            if self.replay_segment_index(approach_point[:2]) == gate_segment:
                return approach_point
            approach_distance *= 0.7

        return gate_point.copy()

    def replay_target_distance(self, target, current_position):
        target_type = (
            self.replay_trajectory_types[self.replay_traj_index]
            if self.replay_trajectory_types
            else None
        )
        if target_type in ("approach", "gate", "exit", "entry"):
            return float(np.linalg.norm(target[:2] - current_position[:2]))
        return float(np.linalg.norm(target - current_position))

    def replay_target_reached(self, target, current_position, target_type):
        xy_distance = float(np.linalg.norm(target[:2] - current_position[:2]))
        if target_type == "gate":
            z_distance = abs(float(target[2] - current_position[2]))
            if self.replay_trajectory_dirs:
                direction = np.array(self.replay_trajectory_dirs[self.replay_traj_index][:2], dtype=float)
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-6:
                    direction = direction / direction_norm
                forward_progress = float(np.dot(current_position[:2] - target[:2], direction))
                lateral_error = float(
                    np.linalg.norm((current_position[:2] - target[:2]) - forward_progress * direction)
                )
            else:
                forward_progress = 0.0
                lateral_error = xy_distance
            cross_margin = self.replay_gate_cross_margin
            if self.is_final_replay_gate_target():
                cross_margin = self.replay_final_gate_cross_margin
            return (
                lateral_error < self.replay_gate_tol
                and z_distance < self.replay_gate_z_tol
                and forward_progress > cross_margin
                and self.replay_gate_crossing_armed_key == self.replay_target_key()
            )
        if target_type == "approach":
            z_distance = abs(float(target[2] - current_position[2]))
            gate_number = self.replay_gate_number(self.replay_traj_index)
            if gate_number is not None and self.replay_segment_index(current_position[:2]) != gate_number + 1:
                return False
            return xy_distance < self.replay_approach_tol and z_distance < self.replay_traj_z_tol

        switch_distance = self.replay_current_tolerance()
        if target_type not in ("approach", "gate"):
            switch_distance = max(switch_distance, self.replay_switch_distance)
        z_distance = abs(float(target[2] - current_position[2]))
        return xy_distance < switch_distance and z_distance < self.replay_traj_z_tol

    def search_command(self, sensor_data, dt):
        if self.scan_center_yaw is None:
            self.scan_center_yaw = sensor_data["yaw"]

        self.scan_phase += dt * 0.8
        current_sine = np.sin(self.scan_phase)
        if self.scan_last_sine != 0.0 and current_sine * self.scan_last_sine < 0.0:
            self.scan_half_swings += 1
        self.scan_last_sine = current_sine

        if self.scan_half_swings >= self.max_scan_half_swings:
            self.state = "ROTATE_LEFT"
            return self.rotate_left_command(sensor_data, dt)

        yaw_target = self.scan_center_yaw + np.deg2rad(70.0) * current_sine
        z_error = self.cruise_altitude - sensor_data["z_global"]
        z_target = sensor_data["z_global"] + np.clip(1.4 * z_error, -0.12, 0.12)

        return [
            sensor_data["x_global"] + self.search_sweep_translation * np.cos(sensor_data["yaw"]),
            sensor_data["y_global"] + self.search_sweep_translation * np.sin(sensor_data["yaw"]),
            z_target,
            yaw_target,
        ]

    def rotate_left_command(self, sensor_data, dt):
        current_yaw = sensor_data["yaw"]
        if self.rotate_left_last_yaw is None:
            self.rotate_left_last_yaw = current_yaw
        else:
            yaw_step = self.wrap_angle(current_yaw - self.rotate_left_last_yaw)
            self.rotate_left_accum += abs(yaw_step)
            self.rotate_left_last_yaw = current_yaw

        if self.rotate_left_accum >= np.pi:
            self.state = "GO_TO_ORBIT"
            return self.go_to_orbit_command(sensor_data)

        yaw_target = sensor_data["yaw"] + np.deg2rad(45.0) * max(dt, 0.05)
        z_error = self.cruise_altitude - sensor_data["z_global"]
        z_target = sensor_data["z_global"] + np.clip(1.2 * z_error, -0.08, 0.08)
        return self.hold_position(sensor_data, altitude=z_target, yaw=yaw_target)

    def go_to_orbit_command(self, sensor_data):
        relative = self.current_xy(sensor_data) - self.arena_center
        norm = np.linalg.norm(relative)
        if norm < 1e-6:
            relative = np.array([0.0, -1.0], dtype=float)
            norm = 1.0

        direction = relative / norm
        target_xy = self.arena_center + self.orbit_radius * direction
        yaw_target = np.arctan2(self.arena_center[1] - target_xy[1], self.arena_center[0] - target_xy[0])

        if np.linalg.norm(target_xy - self.current_xy(sensor_data)) < 0.3:
            self.state = "SEARCH_ORBIT"
            self.orbit_angle = np.arctan2(direction[1], direction[0])
            self.reset_search_pattern(yaw_target)
            return self.search_orbit_command(sensor_data, 0.0)

        return [float(target_xy[0]), float(target_xy[1]), self.cruise_altitude, yaw_target]

    def search_orbit_command(self, sensor_data, dt):
        if self.orbit_angle is None:
            relative = self.current_xy(sensor_data) - self.arena_center
            self.orbit_angle = np.arctan2(relative[1], relative[0])

        self.orbit_angle += 0.22 * dt
        target_xy = self.arena_center + self.orbit_radius * np.array(
            [np.cos(self.orbit_angle), np.sin(self.orbit_angle)],
            dtype=float,
        )
        yaw_target = np.arctan2(self.arena_center[1] - target_xy[1], self.arena_center[0] - target_xy[0])
        return [float(target_xy[0]), float(target_xy[1]), self.cruise_altitude, yaw_target]

    def align_and_approach_command(self, sensor_data, error):
        yaw_correction = -0.6 * error["dx"]
        yaw_target = sensor_data["yaw"] + np.clip(yaw_correction, -np.deg2rad(12), np.deg2rad(12))
        z_target, z_error = self.gate_altitude_target(sensor_data["z_global"], error)

        horizontally_centered = abs(error["dx"]) < 0.12
        vertically_ready = abs(z_error) < 0.12

        if not self.height_aligned and vertically_ready:
            self.height_aligned = True
        if self.height_aligned and abs(z_error) > 0.28:
            self.height_aligned = False

        centered = horizontally_centered and vertically_ready
        very_centered = abs(error["dx"]) < 0.06 and abs(z_error) < 0.08

        if error["area_rel"] > 0.012 and abs(error["dx"]) < 0.2 and abs(z_error) < 0.2:
            self.collect_gate_sample(sensor_data, error)

        if (
            self.require_side_sample_before_store
            and not self.side_sample_used
            and error["area_rel"] > 0.04
            and centered
        ):
            return self.begin_side_sample(sensor_data, z_target, error)

        enough_bearings = len(self.gate_bearing_observations) >= self.min_triangulation_observations
        baseline_ready = self.triangulation_baseline() >= self.min_triangulation_baseline
        if error["area_rel"] > self.triangulation_trigger_area and very_centered and enough_bearings:
            if not baseline_ready and not self.side_sample_used:
                return self.begin_side_sample(sensor_data, z_target, error)
            if not baseline_ready:
                return self.hold_position(sensor_data, altitude=z_target, yaw=yaw_target)
            if not self.maybe_store_gate(sensor_data, error):
                if self.side_sample_used and self.reswipe_attempts < self.max_reswipe_attempts:
                    return self.begin_reswipe_backoff(sensor_data, z_target)
                return self.forward_command(sensor_data, 0.04, z_target, yaw_target=yaw_target)
            return self.start_stored_gate_crossing(sensor_data)

        forward_step = 0.0
        if not self.height_aligned:
            if horizontally_centered and error["area_rel"] > 0.015 and abs(z_error) < 0.22:
                forward_step = 0.08
        elif centered:
            if error["area_rel"] < 0.025:
                forward_step = 0.45
            elif error["area_rel"] < 0.07:
                forward_step = 0.25
            else:
                forward_step = 0.12
        elif horizontally_centered and abs(z_error) < 0.16:
            forward_step = 0.06
        elif abs(error["dx"]) < 0.2 and abs(z_error) < 0.12:
            forward_step = 0.08

        return self.forward_command(sensor_data, forward_step, z_target, yaw_target=yaw_target)

    def pass_through_command(self, sensor_data, dt):
        self.pass_through_timer -= dt
        command = self.forward_command(
            sensor_data,
            self.pass_through_distance,
            self.pass_through_altitude,
            yaw_target=self.pass_through_heading,
            heading=self.pass_through_heading,
        )

        if self.pass_through_timer <= 0.0:
            self.state = "ADVANCE"
            self.advance_timer = self.advance_duration
            self.scan_center_yaw = self.pass_through_heading
            self.reset_gate_localization()
            self.reset_target_switch_tracking()
            self.reset_search_pattern(self.pass_through_heading)

        return command

    def go_to_stored_gate_command(self, sensor_data):
        if self.first_lap_gate_target is None:
            self.state = "ADVANCE"
            self.advance_timer = self.advance_duration
            return self.advance_command(sensor_data, 0.0)

        target = np.array(self.first_lap_gate_target["gate"], dtype=float)
        heading = float(self.first_lap_gate_target["heading"])
        current_position = self.current_xyz(sensor_data)
        xy_error = target[:2] - current_position[:2]
        xy_distance = float(np.linalg.norm(xy_error))
        z_error = float(target[2] - current_position[2])

        if xy_distance < self.first_lap_gate_tol_xy and abs(z_error) < self.first_lap_gate_tol_z:
            self.debug_print(
                f"stored gate reached xy_error={xy_distance:.3f}, z_error={z_error:.3f}; advancing"
            )
            self.pass_through_heading = heading
            self.pass_through_altitude = float(target[2])
            self.first_lap_gate_target = None
            self.reset_target_switch_tracking()
            self.reset_search_pattern(heading)
            self.state = "ADVANCE"
            self.advance_timer = self.advance_duration
            return self.advance_command(sensor_data, 0.0)

        command_xy = target[:2]
        if xy_distance > self.first_lap_gate_xy_step:
            command_xy = current_position[:2] + self.first_lap_gate_xy_step * xy_error / xy_distance

        command_z = current_position[2] + np.clip(
            z_error,
            -self.first_lap_gate_z_step,
            self.first_lap_gate_z_step,
        )
        self.debug_print(
            f"go to stored gate xy_error={xy_distance:.2f}, z_error={z_error:.2f}, "
            f"target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})",
            min_interval=0.5,
        )
        return [float(command_xy[0]), float(command_xy[1]), float(command_z), heading]

    def advance_command(self, sensor_data, dt):
        self.advance_timer -= dt
        command = self.forward_command(
            sensor_data,
            self.advance_distance,
            self.pass_through_altitude,
            yaw_target=self.pass_through_heading,
            heading=self.pass_through_heading,
        )

        if self.advance_timer <= 0.0:
            self.state = "SEARCH"
            self.height_aligned = False
            self.reset_search_pattern(sensor_data["yaw"])
            if len(self.learned_gates) >= self.expected_gate_count:
                self.build_replay_trajectory(
                    entry_position=[
                        sensor_data["x_global"],
                        sensor_data["y_global"],
                        self.pass_through_altitude,
                    ]
                )
                self.state = "REPLAY"
            elif self.learned_gates:
                self.state = "GO_TO_FIRST_LAP_CONE"
                self.first_lap_cone_target = self.current_angle_radius_target(
                    sensor_data,
                    self.pass_through_altitude,
                )
                return self.go_to_first_lap_cone_command(sensor_data)

        return command

    def go_to_first_lap_cone_command(self, sensor_data):
        if self.first_lap_cone_target is None:
            self.first_lap_cone_target = self.current_angle_radius_target(
                sensor_data,
                self.pass_through_altitude,
            )

        current_xy = self.current_xy(sensor_data)
        target = self.first_lap_cone_target
        distance = np.linalg.norm(target[:2] - current_xy)

        if distance < self.first_lap_cone_tol:
            self.first_lap_cone_target = None
            self.state = "SEARCH"
            self.height_aligned = False
            self.reset_search_pattern(float(target[3]))
            return self.search_command(sensor_data, 0.0)

        command_xy = self.step_toward(current_xy, target[:2], self.first_lap_cone_step)
        return [
            float(command_xy[0]),
            float(command_xy[1]),
            float(target[2]),
            sensor_data["yaw"],
        ]

    def replay_command(self, sensor_data):
        if not self.replay_trajectory:
            self.state = "SEARCH"
            return self.search_command(sensor_data, 0.0)

        current_position = self.current_xyz(sensor_data)

        while True:
            target = np.array(self.replay_trajectory[self.replay_traj_index], dtype=float)
            target_type = (
                self.replay_trajectory_types[self.replay_traj_index]
                if self.replay_trajectory_types
                else None
            )
            if not self.replay_target_reached(target, current_position, target_type):
                break

            self.replay_traj_index += 1
            if self.replay_traj_index >= len(self.replay_trajectory):
                self.replay_traj_index = 0
                self.replay_lap += 1
                if self.replay_lap >= self.total_replay_laps:
                    # The scorer completes a lap only when the drone moves from
                    # segment 5 back into segment 0. Keep following the replay
                    # wrap toward gate 0 so the final lap is credited before the
                    # controller returns home.
                    self.replay_lap = self.total_replay_laps

        target = np.array(self.replay_trajectory[self.replay_traj_index], dtype=float)
        target_type = (
            self.replay_trajectory_types[self.replay_traj_index]
            if self.replay_trajectory_types
            else None
        )
        if (
            self.replay_debug
            and (
                self.replay_traj_index != self.last_printed_replay_index
                or target_type != self.last_printed_replay_type
            )
        ):
            gate_number = self.replay_gate_number(self.replay_traj_index)
            print(
                "[assignment replay] "
                f"lap={self.replay_lap} index={self.replay_traj_index} gate={gate_number} "
                f"type={target_type} target=({target[0]:.2f},{target[1]:.2f},{target[2]:.2f}) "
                f"pos=({current_position[0]:.2f},{current_position[1]:.2f},{current_position[2]:.2f})",
                flush=True,
            )
            self.last_printed_replay_index = self.replay_traj_index
            self.last_printed_replay_type = target_type
        distance = self.replay_target_distance(target, current_position)
        command_target = target
        target_key = self.replay_target_key()
        if target_type != "gate" or self.replay_gate_crossing_armed_key != target_key:
            self.replay_gate_crossing_armed_key = None

        if target_type == "gate" and self.replay_trajectory_dirs:
            direction_xy = np.array(self.replay_trajectory_dirs[self.replay_traj_index][:2], dtype=float)
            direction_norm = np.linalg.norm(direction_xy)
            if direction_norm > 1e-6:
                direction = np.array(
                    [
                        direction_xy[0] / direction_norm,
                        direction_xy[1] / direction_norm,
                        0.0,
                    ],
                    dtype=float,
                )
                cross_command = self.replay_gate_cross_command
                if self.is_final_replay_gate_target():
                    cross_command = self.replay_final_gate_cross_command
                forward_progress = float(np.dot(current_position[:2] - target[:2], direction[:2]))
                lateral_error = float(
                    np.linalg.norm((current_position[:2] - target[:2]) - forward_progress * direction[:2])
                )
                z_distance = abs(float(target[2] - current_position[2]))
                if z_distance > self.replay_gate_z_tol or lateral_error > self.replay_gate_tol:
                    self.replay_gate_crossing_armed_key = None
                    command_target = target - self.replay_gate_align_hold_distance * direction
                elif forward_progress > 0.02 and self.replay_gate_crossing_armed_key != target_key:
                    command_target = target - self.replay_gate_align_hold_distance * direction
                else:
                    self.replay_gate_crossing_armed_key = target_key
                    command_target = target + cross_command * direction

        is_final_replay_target = (
            self.replay_lap == self.total_replay_laps - 1
            and self.replay_traj_index == len(self.replay_trajectory) - 1
        )
        if (
            target_type not in ("approach", "gate", "exit")
            and not is_final_replay_target
            and len(self.replay_trajectory) > 1
        ):
            next_index = (self.replay_traj_index + 1) % len(self.replay_trajectory)
            next_target = np.array(self.replay_trajectory[next_index], dtype=float)
            segment = next_target - target
            segment_norm = np.linalg.norm(segment)
            if segment_norm > 1e-6:
                lookahead_ratio = np.clip(1.0 - distance / self.replay_lookahead_distance, 0.0, 1.0)
                along_segment = min(self.replay_segment_lookahead, segment_norm)
                command_target = target + lookahead_ratio * along_segment * segment / segment_norm

        command_xy = command_target[:2]
        command_delta_xy = command_xy - current_position[:2]
        command_distance_xy = np.linalg.norm(command_delta_xy)
        if command_distance_xy > self.replay_xy_step:
            command_xy = current_position[:2] + self.replay_xy_step * command_delta_xy / command_distance_xy

        z_error = float(command_target[2] - current_position[2])
        command_z = current_position[2] + np.clip(z_error, -self.replay_z_step, self.replay_z_step)
        if (
            self.replay_debug
            and target_type == "gate"
            and self.replay_gate_number(self.replay_traj_index) == 0
            and self.time_since_start - self.last_replay_gate0_debug_time > 0.25
        ):
            direction = np.array(self.replay_trajectory_dirs[self.replay_traj_index][:2], dtype=float)
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-6:
                direction = direction / direction_norm
            forward_progress = float(np.dot(current_position[:2] - target[:2], direction))
            lateral_error = float(
                np.linalg.norm((current_position[:2] - target[:2]) - forward_progress * direction)
            )
            print(
                "[assignment replay] gate0 gate-target "
                f"xy_error={distance:.3f} lateral_error={lateral_error:.3f} z_error={z_error:.3f} "
                f"forward_progress={forward_progress:.3f} "
                f"command=({command_xy[0]:.2f},{command_xy[1]:.2f},{command_z:.2f})",
                flush=True,
            )
            self.last_replay_gate0_debug_time = self.time_since_start
        return [float(command_xy[0]), float(command_xy[1]), float(command_z), sensor_data["yaw"]]

    def return_home_command(self, sensor_data):
        dx = self.home_setpoint[0] - sensor_data["x_global"]
        dy = self.home_setpoint[1] - sensor_data["y_global"]
        dz = self.home_setpoint[2] - sensor_data["z_global"]

        if np.linalg.norm([dx, dy, dz]) < 0.25:
            self.state = "WAIT"
            return self.wait_command()

        yaw_target = np.arctan2(dy, dx) if abs(dx) + abs(dy) > 1e-6 else self.home_setpoint[3]
        return [
            self.home_setpoint[0],
            self.home_setpoint[1],
            self.home_setpoint[2],
            yaw_target,
        ]

    def wait_command(self):
        return list(self.home_setpoint)

    def compute_command(self, sensor_data, camera_data, dt):
        self.time_since_start += dt

        if self.scan_center_yaw is None:
            self.scan_center_yaw = sensor_data["yaw"]

        if self.state == "TAKEOFF":
            if sensor_data["z_global"] < self.cruise_altitude - 0.05:
                return self.hold_position(sensor_data, altitude=self.cruise_altitude, yaw=sensor_data["yaw"])
            self.state = "SEARCH"

        if self.state == "PASS_THROUGH":
            return self.pass_through_command(sensor_data, dt)
        if self.state == "ADVANCE":
            return self.advance_command(sensor_data, dt)
        if self.state == "GO_TO_FIRST_LAP_CONE":
            return self.go_to_first_lap_cone_command(sensor_data)
        if self.state == "SIDE_SAMPLE":
            return self.side_sample_command(sensor_data, camera_data, dt)
        if self.state == "RESWIPE_BACKOFF":
            return self.reswipe_backoff_command(sensor_data, camera_data)
        if self.state == "GO_TO_STORED_GATE":
            return self.go_to_stored_gate_command(sensor_data)
        if self.state == "REPLAY":
            return self.replay_command(sensor_data)
        if self.state == "RETURN_HOME":
            return self.return_home_command(sensor_data)
        if self.state == "WAIT":
            return self.wait_command()
        if self.state == "GO_TO_ORBIT":
            return self.go_to_orbit_command(sensor_data)

        if self.state == "SEARCH_ORBIT":
            error = self.detect_gate(camera_data)
            if error is not None:
                return self.handle_detected_gate(sensor_data, error)
            return self.search_orbit_command(sensor_data, dt)

        if self.state == "ROTATE_LEFT":
            error = self.detect_gate(camera_data)
            if error is not None:
                return self.handle_detected_gate(sensor_data, error)
            return self.rotate_left_command(sensor_data, dt)

        error = self.detect_gate(camera_data)
        if error is not None:
            return self.handle_detected_gate(sensor_data, error)

        recently_seen = (self.time_since_start - self.last_seen_time) < 1.0 and self.last_seen_yaw is not None
        if recently_seen:
            forward_step = 0.12 if self.height_aligned else 0.0
            return self.forward_command(
                sensor_data,
                forward_step,
                sensor_data["z_global"],
                yaw_target=self.last_seen_yaw,
            )

        self.state = "SEARCH"
        self.height_aligned = False
        return self.search_command(sensor_data, dt)


_controller = MyAssignment()


def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
