import cv2
import mediapipe as mp

mp_manos = mp.solutions.hands
manos = mp_manos.Hands()
mp_dibujo = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("✅ Cámara iniciada. Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer el frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = manos.process(frame_rgb)

    if resultado.multi_hand_landmarks:
        print("✋ Mano detectada.")
        for mano in resultado.multi_hand_landmarks:
            mp_dibujo.draw_landmarks(frame, mano, mp_manos.HAND_CONNECTIONS)

    cv2.imshow("Detección de manos", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("🚪 Saliendo...")
        break

cap.release()
cv2.destroyAllWindows()
