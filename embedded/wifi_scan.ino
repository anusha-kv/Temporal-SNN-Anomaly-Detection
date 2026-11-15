
#include "snn_model.h"

const float FIRING_THRESHOLD = 1.0f;
const float INPUT_COUNT_NORMALIZER = 5.0f;

const int VOLTAGE_PIN = 34;
const int CURRENT_PIN = 35;

const int GREEN_LED = 25;
const int RED_LED = 26;

float voltageBins[TIME_BINS];
float currentBins[TIME_BINS];

void testLEDs() {
  Serial.println("Testing LEDs...");

  digitalWrite(GREEN_LED, HIGH);
  digitalWrite(RED_LED, LOW);
  delay(1000);

  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, HIGH);
  delay(1000);

  digitalWrite(RED_LED, LOW);
  Serial.println("LED test completed.");
}

void readPotentiometers() {
  for (int t = 0; t < TIME_BINS; t++) {
    int voltageRaw = analogRead(VOLTAGE_PIN);
    int currentRaw = analogRead(CURRENT_PIN);

    voltageBins[t] = (voltageRaw / 4095.0f) * 5.0f;
    currentBins[t] = (currentRaw / 4095.0f) * 5.0f;

    delay(100);
  }
}

void extractFeatures(float features[]) {
  float membrane[HIDDEN_NEURONS] = {0};

  for (int t = 0; t < TIME_BINS; t++) {
    float voltage =
      voltageBins[t] / INPUT_COUNT_NORMALIZER * INPUT_SCALE;

    float current =
      currentBins[t] / INPUT_COUNT_NORMALIZER * INPUT_SCALE;

    for (int n = 0; n < HIDDEN_NEURONS; n++) {
      float inputCurrent =
        voltage * W_INPUT[n] +
        current * W_INPUT[HIDDEN_NEURONS + n];

      membrane[n] = DECAY * membrane[n] + inputCurrent;

      if (membrane[n] >= FIRING_THRESHOLD) {
        features[t * HIDDEN_NEURONS + n] = 1.0f;
        membrane[n] = 0.0f;
      } else {
        features[t * HIDDEN_NEURONS + n] = 0.0f;
      }
    }

    features[640 + t * 2] = voltage;
    features[640 + t * 2 + 1] = current;
  }
}

float classifySignal() {
  float features[FEATURE_COUNT] = {0};

  extractFeatures(features);

  float score = INTERCEPT;

  for (int i = 0; i < FEATURE_COUNT; i++) {
    float stdValue = X_STD[i];

    if (stdValue > -0.000001f && stdValue < 0.000001f) {
      stdValue = 1.0f;
    }

    float normalized =
      (features[i] - X_MEAN[i]) / stdValue;

    score += normalized * W_OUTPUT[i];
  }

  return score;
}

void setup() {
  Serial.begin(115200);

  analogReadResolution(12);

  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);

  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, LOW);

  Serial.println("ESP32 SNN Anomaly Detection");
  Serial.println("System initialized!");

  testLEDs();
}

void loop() {
  readPotentiometers();

  float score = classifySignal();

  Serial.print("Anomaly score: ");
  Serial.println(score, 6);

  if (score >= THRESHOLD) {
    Serial.println("Result: ABNORMAL SIGNAL");

    digitalWrite(RED_LED, HIGH);
    digitalWrite(GREEN_LED, LOW);
  } else {
    Serial.println("Result: NORMAL SIGNAL");

    digitalWrite(GREEN_LED, HIGH);
    digitalWrite(RED_LED, LOW);
  }

  Serial.println("----------------------");

  delay(1000);
}
