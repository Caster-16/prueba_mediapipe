import cv2
import mediapipe as mp
import time

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
        for mano in resultado.multi_hand_landmarks:
            mp_dibujo.draw_landmarks(frame, mano, mp_manos.HAND_CONNECTIONS)

            # Obtener coordenadas del punto 8 (dedo índice)
            h, w, _ = frame.shape
            punto_8 = mano.landmark[8]
            punto_6 = mano.landmark[6]
            cx, cy = int(punto_8.x * w), int(punto_8.y * h)

            cv2.putText(frame, "8", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            print(f"🟡 ID 8 (índice): ({cx}, {cy})")
            time.sleep(0.1)

            # Verifica si el dedo índice está arriba
            if punto_8.y < punto_6.y:
                cv2.putText(frame, "Indice arriba", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            print("✋ Mano detectada.")
            cv2.putText(frame, "¡Mano detectada!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            break  # ← Solo procesar una mano por frame  

    cv2.imshow("Deteccion de manos", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        print("🚪 Saliendo...")
        break

cap.release()
cv2.destroyAllWindows()
