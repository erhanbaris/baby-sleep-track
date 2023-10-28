import os
from typing import Any
import cv2 as cv
import mediapipe as mp
import time
import math
from datetime import datetime
from requests import get, post
from enum import Enum
import operator
from abc import ABC, abstractmethod
import logging

IN_DOCKER = os.environ.get('IN_DOCKER', False)
STATE_CHANGE_TIMEOUT = 30
SLEEP_TIMEOUT = 10 * 60
ENTITY_NAME = "baby.sleep_time"

if not IN_DOCKER:
    import utils

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# constants
FONTS = cv.FONT_HERSHEY_COMPLEX

LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


logging.basicConfig(
    format='[%(asctime)s][%(levelname)-8s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


class State(Enum):
    IDLE = 0
    AWAKE = 1
    SLEEPING = 2

    def __str__(self) -> str:
        if self == State.IDLE:
            return "Idle"

        elif self == State.AWAKE:
            return "Awake"

        elif self == State.SLEEPING:
            return "Sleeping"

        else:
            return "N/A"


class StateChanger(ABC):
    @abstractmethod
    def change(self, state: State, engine):
        pass


class StatusDispacher:
    def __init__(self) -> None:
        self._dispachers = []

    def add(self, dispatcher):
        self._dispachers.append(dispatcher)

    def change(self, status: State):
        for dispatcher in self._dispachers:
            dispatcher(status)


def log_status(status: State):
    logging.info(f"State changed to {str(status)}")


class HomeAssistantDispacher:
    def __init__(self) -> None:
        self.integration_code = "INTEGRATION_CODE"
        self.sleeping_entity = f"https://192.168.1.122/api/states/{ENTITY_NAME}"
        self.sleep_time_entity = f"https://192.168.1.122/api/states/{ENTITY_NAME}"

        self.headers = {
            "Authorization": "Bearer " + self.integration_code,
            "content-type": "application/json",
        }

    def __call__(self, status: State) -> Any:
        attributes = {
            'last_sleeping_time': 0,
            'last_awake_time': 0,
            'last_idle_time': 0
        }

        # Read and configure last times
        result = get(self.sleep_time_entity, headers=self.headers)
        if result.ok:
            old_attributes = result.json().get("attributes", {})
            attributes['last_sleeping_time'] = old_attributes.get('last_sleeping_time', 0)
            attributes['last_awake_time'] = old_attributes.get('last_awake_time', 0)
            attributes['last_idle_time'] = old_attributes.get('last_idle_time', 0)

        sleeping_query = {'state': str(status)}
        start_time_query = {
            'state': 'N/A',
            'attributes': attributes
        }

        # Send state
        result = post(self.sleeping_entity,
                      headers=self.headers,
                      json=sleeping_query)

        if not result.ok:
            logging.error(f"Home Assistant did not accept our request: {result}")

        # Update attributes
        if status is State.SLEEPING:

            if time.time() - start_time_query['attributes']['last_sleeping_time'] < SLEEP_TIMEOUT:
                # Still count in sleeping
                start_time_query['state'] = datetime.fromtimestamp(start_time_query['attributes']['last_sleeping_time']).strftime("%H:%M:%S")
            else:
                # New sleeping session
                start_time_query['state'] = datetime.now().strftime("%H:%M:%S")
                start_time_query['attributes']['last_sleeping_time'] = time.time()

        elif status is State.AWAKE:
            start_time_query['attributes']['last_awake_time'] = time.time()

        elif status is State.IDLE:
            start_time_query['attributes']['last_idle_time'] = time.time()

        # Send times
        result = post(self.sleep_time_entity,
                      headers=self.headers,
                      json=start_time_query)
        if not result.ok:
            logging.error(f"Home Assistant did not accept our request: {result}")


class RtspCameraResource:
    def __init__(self) -> None:
        self.camera = None
        self.url = None

    def is_open(self):
        return self.camera.isOpened()

    def configure(self, url):
        self.camera = cv.VideoCapture(url, cv.CAP_FFMPEG)
        self.url = url

    def read(self):
        if self.camera is None:
            self.configure(self.url)

        ret, frame = self.camera.read()
        if not ret:
            self.camera.release()
            self.camera = None
            cv.destroyAllWindows()
            return False, None
        return True, frame


class LocalCameraResource:
    def __init__(self) -> None:
        self.camera = None

    def is_open(self):
        return self.camera.isOpened()

    def configure(self, camera_id):
        self.camera = cv.VideoCapture(camera_id)

    def read(self):
        ret, frame = self.camera.read()
        if not ret:
            return False, None
        return True, frame


class Engine:
    ROTATIONS = [cv.ROTATE_90_CLOCKWISE, cv.ROTATE_90_COUNTERCLOCKWISE, cv.ROTATE_180]

    def __init__(self) -> None:
        self.frame_counter = 0
        self.fps = 0.0
        self.video_resource = None
        self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.3)
        self.start_time = time.time()
        self.rotation = None
        self.current_ratio = 0.0
        self.current_state = State.IDLE
        self.state_dispatcher = None
        self.state_checker = None

    def set_video_resource(self, resource):
        self.video_resource = resource

    def set_state_dispatcher(self, dispatcher):
        self.state_dispatcher = dispatcher

    def set_state_checker(self, checker):
        self.state_checker = checker

    def draw_face_mesh(self, frame, face_landmarks):
        mp_drawing.draw_landmarks(image=frame, 
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        mp_drawing.draw_landmarks(image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None, 
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    def process_face(self, rgb_frame, results):
        self.draw_face_mesh(rgb_frame, results.multi_face_landmarks[0])

        mesh_coords = self.landmarks_detection(rgb_frame, results, False)
        self.current_ratio = self.blink_ratio(rgb_frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

        if self.current_ratio > 4.5:
            self.state_checker.check(State.SLEEPING, self)
        else:
            self.state_checker.check(State.AWAKE, self)

    def try_to_idle(self):
        self.state_checker.check(State.IDLE, self)
        self.rotation = None

        for rotation_code in self.ROTATIONS:
            if self.check_for_face(rotation_code):
                self.rotation = rotation_code

    def check_for_face(self, rotate):
        frame = cv.rotate(self.edited_frame, rotate)
        result = self.face_mesh.process(frame)

        if result.multi_face_landmarks:
            return True
        return False

    def landmarks_detection(self, img, results, draw=False):
        img_height, img_width = img.shape[:2]
        mesh_coord = [
            (int(point.x * img_width), int(point.y * img_height))
            for point in results.multi_face_landmarks[0].landmark
        ]
        if draw:
            [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

        return mesh_coord

    def euclaidean_distance(self, point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
        return distance

    def blink_ratio(self, img, landmarks, right_indices, left_indices):
        # Right eyes
        # horizontal line
        rh_right = landmarks[right_indices[0]]
        rh_left  = landmarks[right_indices[8]]
        # vertical line
        rv_top    = landmarks[right_indices[12]]
        rv_bottom = landmarks[right_indices[4]]

        # LEFT_EYE
        # horizontal line
        lh_right = landmarks[left_indices[0]]
        lh_left  = landmarks[left_indices[8]]

        # vertical line
        lv_top    = landmarks[left_indices[12]]
        lv_bottom = landmarks[left_indices[4]]

        rhDistance = self.euclaidean_distance(rh_right, rh_left)
        rvDistance = self.euclaidean_distance(rv_top, rv_bottom)

        lvDistance = self.euclaidean_distance(lv_top, lv_bottom)
        lhDistance = self.euclaidean_distance(lh_right, lh_left)

        reRatio = rhDistance / rvDistance
        leRatio = lhDistance / lvDistance

        ratio = (reRatio + leRatio) / 2
        return ratio

    def is_open(self):
        return self.video_resource.is_open()

    def process(self):
        self.frame_counter += 1
        status, frame = self.video_resource.read()
        if not status:
            if self.current_state != State.IDLE:
                self.state_dispatcher.change(State.IDLE)
            logging.error("Video resource not ready")
            return False
        
        # Prepare image
        self.original_frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        if self.rotation:
            self.original_frame = cv.rotate(self.original_frame, self.rotation)

        self.edited_frame = cv.cvtColor(self.original_frame, cv.COLOR_RGB2BGR)

        results = self.face_mesh.process(self.edited_frame)

        if results.multi_face_landmarks:
            self.process_face(self.edited_frame, results)
        else:
            self.try_to_idle()

        # calculating  frame per seconds FPS
        end_time = time.time() - self.start_time
        self.fps = self.frame_counter / end_time

        return True


class CounterBasedStateChanger(StateChanger):
    def __init__(self) -> None:
        self.counter = 0
        self.state_counts = {
            State.IDLE: 0,
            State.AWAKE: 0,
            State.SLEEPING: 0
        }

    def change(self, state: State, engine: Engine):
        self.counter = self.counter + 1
        self.state_counts[state] = self.state_counts[state] + 1

        if self.counter >= engine.fps * STATE_CHANGE_TIMEOUT:
            self.counter = 0
            most_active_state = max(self.state_counts.items(), key=operator.itemgetter(1))[0]
            self.state_counts[State.AWAKE] = 0
            self.state_counts[State.SLEEPING] = 0
            self.state_counts[State.IDLE] = 0

            if engine.current_state is state:
                return

            if engine.current_state != most_active_state:
                engine.state_dispatcher.change(most_active_state)
                engine.current_state = most_active_state


class StateChecker:
    def __init__(self) -> None:
        self._checkers = []

    def add(self, checker):
        self._checkers.append(checker)

    def check(self, status: State, engine: Engine):
        for checker in self._checkers:
            checker.change(status, engine)


class EngineManager:
    def __init__(self) -> None:
        self.engine = None # type: Engine

    def set_engine(self, engine):
        self.engine = engine

    def start(self):
        logging.info("Starting, please wait")

        edit_mode = False

        while True:
            if not self.engine.process():
                logging.warning("Waiting 1.0 seconds")
                time.sleep(1.0)
                continue
            
            if not IN_DOCKER:
                if edit_mode:
                    frame = self.engine.edited_frame
                else:
                    frame = self.engine.original_frame

                frame = utils.colorBackgroundText(frame, f"Ratio : {round(self.engine.current_ratio, 2)}", FONTS, 1.0, (30, 100), 2, utils.PINK, utils.YELLOW)
                frame = utils.textWithBackground(frame, f"FPS: {round(self.engine.fps, 1)}", FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
                frame = utils.colorBackgroundText(frame, f"State: {str(self.engine.current_state)}", FONTS, 1.0, (30, 150), 2, utils.YELLOW, pad_x=6, pad_y=6)

                cv.imshow("frame", frame)
                key = cv.waitKey(2)
                if key == ord("q") or key == ord("Q"):
                    return
                elif key == ord("e") or key == ord("E"):
                    edit_mode = not edit_mode


home_assistant_dispatcher = HomeAssistantDispacher()

status_dispatcher = StatusDispacher()
status_dispatcher.add(home_assistant_dispatcher)
status_dispatcher.add(log_status)

camera_resource = LocalCameraResource()
camera_resource.configure("rtsp://admin:password@192.168.1.123:8554/Streaming/Channels/102")

state_checker = StateChecker()
state_checker.add(CounterBasedStateChanger())

engine = Engine()
engine.set_state_dispatcher(status_dispatcher)
engine.set_video_resource(camera_resource)
engine.set_state_checker(state_checker)

manager = EngineManager()
manager.set_engine(engine)
manager.start()
