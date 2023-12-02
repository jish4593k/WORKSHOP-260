
import cv2
import turtle
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model


model = load_model('your_model.h5')


screen = turtle.Screen()
screen.title("Live Streaming with Turtle and Neural Network")
screen.setup(width=800, height=600)
screen.tracer(0)  # Turn off automatic screen updates

# Create a Turtle for drawing predictions
prediction_turtle = turtle.Turtle()
prediction_turtle.hideturtle()
prediction_turtle.penup()
prediction_turtle.goto(-300, 200)


cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()

    # Preprocess the frame for neural network input (replace with your preprocessing)
    processed_frame = cv2.resize(frame, (224, 224))
    processed_frame = processed_frame / 255.0
    processed_frame = np.expand_dims(processed_frame, axis=0)

    
    prediction = model.predict(processed_frame)[0]

    
    prediction_text = f"Prediction: {np.argmax(prediction)}"
    prediction_turtle.clear()
    prediction_turtle.write(prediction_text, font=("Arial", 16, "normal"))

  
    cv2.imshow('Live Streaming', frame)

   
    screen.update()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
