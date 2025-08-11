import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import threading
import tkinter as tk
import matplotlib.pyplot as plt


mouse_control_enabled = False
dragging = False
screen_w, screen_h = pyautogui.size()
pinch_threshold = 40
# Accuracy tracking variables
total_frames = 0
successful_detections = 0
pinch_detections = 0
palm_open_detections = 0



def get_landmark_positions(hand_landmarks, w, h):
    x1 = int(hand_landmarks.landmark[8].x * w)
    y1 = int(hand_landmarks.landmark[8].y * h)
    x2 = int(hand_landmarks.landmark[4].x * w)
    y2 = int(hand_landmarks.landmark[4].y * h)
    return (x1, y1), (x2, y2)


def is_pinching(index_tip, thumb_tip):
    distance = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
    return distance < pinch_threshold


def move_cursor(index_tip, frame_shape):
    h, w, _ = frame_shape
    screen_x = np.interp(index_tip[0], [0, w], [0, screen_w])
    screen_y = np.interp(index_tip[1], [0, h], [0, screen_h])
    pyautogui.moveTo(screen_x, screen_y)


def start_drag(frame, pos):
    global dragging
    dragging = True
    pyautogui.mouseDown()
    cv2.putText(frame, 'DRAG START', (pos[0], pos[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


def stop_drag(frame, pos):
    global dragging
    dragging = False
    pyautogui.mouseUp()
    cv2.putText(frame, 'DRAG END', (pos[0], pos[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


def virtual_mouse_loop():
    global dragging, mouse_control_enabled, total_frames, successful_detections, pinch_detections, palm_open_detections

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    def is_palm_open(hand_landmarks):
      finger_tips = [4, 8, 12, 16, 20]
      finger_joints = [3, 6, 10, 14, 18]

      fingers_open = 0
      for tip_id, joint_id in zip(finger_tips, finger_joints):
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[joint_id].y:
            fingers_open += 1

      return fingers_open == 5


    while True:
        success, frame = cap.read()
        if not success:
            break

        total_frames += 1

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            successful_detections += 1
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if is_palm_open(hand_landmarks):
                    palm_open_detections += 1
                    cv2.putText(frame, 'FROZEN', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    continue
                index_tip, thumb_tip = get_landmark_positions(hand_landmarks, w, h)

                cv2.circle(frame, index_tip, 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(frame, thumb_tip, 10, (0, 255, 0), cv2.FILLED)
                cv2.line(frame, index_tip, thumb_tip, (0, 255, 255), 2)

                if mouse_control_enabled:
                    move_cursor(index_tip, frame.shape)

                    if is_pinching(index_tip, thumb_tip):
                        pinch_detections += 1
                        if not dragging:
                            start_drag(frame, index_tip)
                    else:
                        if dragging:
                            stop_drag(frame, index_tip)

        status = "ON" if mouse_control_enabled else "OFF"
        cv2.putText(frame, f'Mouse Control: {status}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if mouse_control_enabled else (0, 0, 255), 1)

        cv2.imshow("AI Virtual Mouse - GUI Toggle", frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    

    cap.release()
    cv2.destroyAllWindows()
    show_accuracy_graph()

def show_accuracy_graph():
    detection_accuracy = (successful_detections / total_frames) * 100 if total_frames else 0
    pinch_accuracy = (pinch_detections / successful_detections) * 100 if successful_detections else 0
    palm_accuracy = (palm_open_detections / successful_detections) * 100 if successful_detections else 0

    print(f"\n--- AI Virtual Mouse Accuracy Report ---")
    print(f"Overall Hand Detection Accuracy: {detection_accuracy:.2f}%")
    print(f"Pinch Gesture Accuracy: {pinch_accuracy:.2f}%")
    print(f"Palm Open Gesture Accuracy: {palm_accuracy:.2f}%\n")

    labels = ['Hand Detection', 'Pinch', 'Palm Open']
    accuracies = [detection_accuracy, pinch_accuracy, palm_accuracy]

    plt.bar(labels, accuracies, color=['blue', 'green', 'orange'])
    plt.ylabel("Accuracy (%)")
    plt.title("AI Virtual Mouse Sensitivity & Accuracy")
    plt.ylim(0, 100)
    plt.show()



def run_gui():
    def toggle_control():
        global mouse_control_enabled
        mouse_control_enabled = not mouse_control_enabled
        toggle_button.config(text="Stop Mouse Control" if mouse_control_enabled else "Start Mouse Control")

    def quit_program():
        root.destroy()
        cv2.destroyAllWindows()
        exit(0)

    root = tk.Tk()
    root.title("Virtual Mouse Control")
    root.geometry("300x150")

    tk.Label(root, text="AI Virtual Mouse", font=("Arial", 16)).pack(pady=10)
    toggle_button = tk.Button(root, text="Start Mouse Control", font=("Arial", 12), command=toggle_control)
    toggle_button.pack(pady=10)
    tk.Button(root, text="Quit", font=("Arial", 12), command=quit_program).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    threading.Thread(target=virtual_mouse_loop, daemon=True).start()
    run_gui()
