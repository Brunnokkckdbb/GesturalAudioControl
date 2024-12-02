import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import numpy as np
import time
import keyboard  # Para enviar el comando Play/Pause y Next

# Inicialización de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inicialización de Pycaw para controlar el volumen
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
min_volume, max_volume, _ = volume.GetVolumeRange()

# Variable para mostrar o no la ventana de la cámara
show_camera = True  # Cambiar a False para ocultar la ventana de la cámara

# Función para calcular el ángulo de la muñeca
def calculate_wrist_angle(landmarks):
    wrist = landmarks[0]
    index_base = landmarks[5]
    pinky_base = landmarks[17]
    ref_vector = np.array([pinky_base.x - index_base.x, pinky_base.y - index_base.y])
    wrist_vector = np.array([wrist.x - index_base.x, wrist.y - index_base.y])
    dot_product = np.dot(ref_vector, wrist_vector)
    mag_ref = np.linalg.norm(ref_vector)
    mag_wrist = np.linalg.norm(wrist_vector)
    angle = np.arccos(dot_product / (mag_ref * mag_wrist + 1e-8))  # Evitar división por cero
    return np.degrees(angle)

# Función para verificar si los dedos están abiertos
def are_fingers_open(landmarks):
    tips = [8, 12, 16, 20]  # Índice, medio, anular, meñique
    bases = [6, 10, 14, 18]  # Articulaciones base de los dedos
    for tip, base in zip(tips, bases):
        if landmarks[tip].y > landmarks[base].y:  # Comprobar si la punta del dedo está por debajo de la articulación base
            return False
    return True

# Función para detectar gesto de Play/Pause
def is_play_pause_gesture(landmarks):
    tips = [8, 12, 16, 20]  # Índices de las puntas de los dedos
    bases = [6, 10, 14, 18]  # Índices de las articulaciones base
    if (landmarks[tips[0]].y < landmarks[bases[0]].y and  # Índice levantado
        landmarks[tips[1]].y < landmarks[bases[1]].y and  # Medio levantado
        landmarks[tips[2]].y < landmarks[bases[2]].y and  # Anular levantado
        landmarks[tips[3]].y > landmarks[bases[3]].y):    # Meñique abajo
        return True
    return False

# Función para enviar comando Play/Pause
def send_play_pause_command():
    keyboard.press_and_release('play/pause media')

# Función para detectar gesto de Next
def is_next_gesture(landmarks):
    """
    Detecta si solo los dedos índice y meñique están levantados
    mientras que los demás dedos (medio, anular y pulgar) están abajo.
    """
    tips = [8, 12, 16, 20]  # Índices de las puntas de los dedos
    bases = [6, 10, 14, 18]  # Índices de las articulaciones base

    # Comprobar que índice y meñique están levantados
    if (landmarks[tips[0]].y < landmarks[bases[0]].y and  # Índice levantado
        landmarks[tips[3]].y < landmarks[bases[3]].y and  # Meñique levantado
        landmarks[tips[1]].y > landmarks[bases[1]].y and  # Medio abajo
        landmarks[tips[2]].y > landmarks[bases[2]].y):    # Anular abajo
        return True
    return False

# Función para enviar comando Next
def send_next_command():
    keyboard.press_and_release('next track')

# Inicia la captura de video
cap = cv2.VideoCapture(0)

last_change_time = 0
change_delay = 0.3  # Tiempo en segundos entre cada cambio de volumen

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            try:
                # Detectar gesto de Next
                if is_next_gesture(hand_landmarks.landmark):
                    send_next_command()
                    time.sleep(0.5)  # Evitar múltiples activaciones consecutivas

                # Detectar gesto de Play/Pause
                if is_play_pause_gesture(hand_landmarks.landmark):
                    send_play_pause_command()
                    time.sleep(0.5)  # Evitar múltiples activaciones consecutivas

                angle = calculate_wrist_angle(hand_landmarks.landmark)
                fingers_open = are_fingers_open(hand_landmarks.landmark)

                # Obtener el volumen actual
                current_volume = volume.GetMasterVolumeLevelScalar() * 100  # Multiplicamos por 100 para obtener el porcentaje

                # Actualizar volumen si los dedos están abiertos
                if fingers_open:
                    current_time = time.time()
                    if current_time - last_change_time > change_delay:
                        if angle > 69:  # Muñeca inclinada hacia abajo
                            volume.SetMasterVolumeLevelScalar(max(0.0, volume.GetMasterVolumeLevelScalar() - 0.05), None)
                            last_change_time = current_time
                        elif angle < 53:  # Muñeca inclinada hacia arriba
                            volume.SetMasterVolumeLevelScalar(min(1.0, volume.GetMasterVolumeLevelScalar() + 0.05), None)
                            last_change_time = current_time

                # Mostrar textos solo si `show_camera` es True
                if show_camera:
                    cv2.putText(frame, f"Angle: {int(angle)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Fingers Open" if fingers_open else "Fingers Closed",
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Volume: {int(current_volume)}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except:
                # Si ocurre un error con los cálculos, simplemente pasa
                pass

    # Mostrar la cámara solo si `show_camera` es True
    if show_camera:
        cv2.imshow("Control de Volumen con la Mano", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Presiona Esc para salir
            break
    else:
        if cv2.waitKey(1) & 0xFF == 27:  # Salir si la cámara está desactivada
            break

cap.release()
cv2.destroyAllWindows()
hands.close()
