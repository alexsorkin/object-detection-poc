using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using System.IO;

namespace MilitaryTargetDetection
{
    /// <summary>
    /// Example Unity component demonstrating real-time detection from camera feed
    /// </summary>
    public class RealtimeDetectionExample : MonoBehaviour
    {
        [Header("Camera Setup")]
        public Camera targetCamera;
        public RenderTexture renderTexture;
        
        [Header("UI Elements")]
        public RawImage previewImage;
        public Text detectionCountText;
        public Text fpsText;
        public Text statusText;
        public Button startButton;
        public Button stopButton;
        public Slider confidenceSlider;
        public Text confidenceText;

        [Header("Detection Visualization")]
        public GameObject detectionBoxPrefab;
        public Transform detectionParent;
        public bool showBoundingBoxes = true;
        public bool showConfidenceText = true;

        [Header("Performance")]
        [Range(1, 30)]
        public int targetFPS = 15;
        public bool enablePerformanceMonitoring = true;

        private MilitaryTargetDetector detector;
        private bool isDetecting = false;
        private Texture2D captureTexture;
        private List<GameObject> activeDetectionBoxes = new List<GameObject>();
        
        // Performance monitoring
        private float lastDetectionTime;
        private Queue<float> frameTimeHistory = new Queue<float>();
        private const int MAX_FRAME_HISTORY = 30;

        void Start()
        {
            InitializeComponents();
            SetupUI();
        }

        void InitializeComponents()
        {
            // Get or create detector
            detector = GetComponent<MilitaryTargetDetector>();
            if (detector == null)
            {
                detector = gameObject.AddComponent<MilitaryTargetDetector>();
            }

            // Setup camera and render texture
            if (targetCamera == null)
                targetCamera = Camera.main;

            if (renderTexture == null)
            {
                renderTexture = new RenderTexture(640, 640, 24);
                renderTexture.Create();
            }

            targetCamera.targetTexture = renderTexture;

            // Create capture texture
            captureTexture = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);

            // Setup events
            detector.OnDetectionComplete += OnDetectionComplete;
            detector.OnError += OnDetectionError;
        }

        void SetupUI()
        {
            if (previewImage != null)
                previewImage.texture = renderTexture;

            if (startButton != null)
                startButton.onClick.AddListener(StartDetection);

            if (stopButton != null)
            {
                stopButton.onClick.AddListener(StopDetection);
                stopButton.interactable = false;
            }

            if (confidenceSlider != null)
            {
                confidenceSlider.onValueChanged.AddListener(OnConfidenceChanged);
                confidenceSlider.value = 0.5f;
                OnConfidenceChanged(0.5f);
            }

            UpdateStatusText("Ready to start detection");
        }

        public void StartDetection()
        {
            if (!detector.IsReady)
            {
                UpdateStatusText("Initializing detector...");
                if (!detector.InitializeDetector())
                {
                    UpdateStatusText("Failed to initialize detector");
                    return;
                }
            }

            isDetecting = true;
            StartCoroutine(DetectionLoop());
            
            if (startButton != null) startButton.interactable = false;
            if (stopButton != null) stopButton.interactable = true;
            
            UpdateStatusText("Detection started");
        }

        public void StopDetection()
        {
            isDetecting = false;
            ClearDetectionBoxes();
            
            if (startButton != null) startButton.interactable = true;
            if (stopButton != null) stopButton.interactable = false;
            
            UpdateStatusText("Detection stopped");
        }

        private IEnumerator DetectionLoop()
        {
            float frameInterval = 1f / targetFPS;
            
            while (isDetecting)
            {
                float frameStart = Time.realtimeSinceStartup;
                
                // Capture frame from camera
                RenderTexture.active = renderTexture;
                captureTexture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
                captureTexture.Apply();
                RenderTexture.active = null;

                // Run detection
                var result = detector.DetectInTexture(captureTexture);
                
                // Update performance metrics
                if (enablePerformanceMonitoring)
                {
                    float frameTime = Time.realtimeSinceStartup - frameStart;
                    UpdatePerformanceMetrics(frameTime);
                }

                // Wait for next frame
                float elapsed = Time.realtimeSinceStartup - frameStart;
                if (elapsed < frameInterval)
                {
                    yield return new WaitForSeconds(frameInterval - elapsed);
                }
                else
                {
                    yield return null; // Skip frame if we're running behind
                }
            }
        }

        private void OnDetectionComplete(MilitaryTargetDetector.DetectionResult result)
        {
            if (!isDetecting) return;

            // Update UI on main thread
            if (detectionCountText != null)
            {
                detectionCountText.text = $"Detections: {result.Count}";
            }

            // Update visual detection boxes
            if (showBoundingBoxes)
            {
                UpdateDetectionBoxes(result);
            }

            lastDetectionTime = Time.realtimeSinceStartup;
        }

        private void OnDetectionError(string error)
        {
            Debug.LogError($"Detection Error: {error}");
            UpdateStatusText($"Error: {error}");
        }

        private void UpdateDetectionBoxes(MilitaryTargetDetector.DetectionResult result)
        {
            // Clear existing boxes
            ClearDetectionBoxes();

            if (detectionBoxPrefab == null || detectionParent == null)
                return;

            // Create new boxes for each detection
            foreach (var detection in result.detections)
            {
                GameObject boxObj = Instantiate(detectionBoxPrefab, detectionParent);
                
                // Position and scale the box
                RectTransform rectTransform = boxObj.GetComponent<RectTransform>();
                if (rectTransform != null)
                {
                    // Convert normalized coordinates to UI coordinates
                    var bbox = detection.boundingBox;
                    float uiWidth = ((RectTransform)detectionParent).rect.width;
                    float uiHeight = ((RectTransform)detectionParent).rect.height;

                    rectTransform.anchorMin = new Vector2(bbox.x, 1 - bbox.y - bbox.height);
                    rectTransform.anchorMax = new Vector2(bbox.x + bbox.width, 1 - bbox.y);
                    rectTransform.offsetMin = Vector2.zero;
                    rectTransform.offsetMax = Vector2.zero;
                }

                // Set box color based on class
                Image boxImage = boxObj.GetComponent<Image>();
                if (boxImage != null)
                {
                    boxImage.color = MilitaryTargetDetector.GetClassColor(detection.targetClass);
                }

                // Set confidence text
                if (showConfidenceText)
                {
                    Text confidenceLabel = boxObj.GetComponentInChildren<Text>();
                    if (confidenceLabel != null)
                    {
                        string className = MilitaryTargetDetector.GetClassName(detection.targetClass);
                        confidenceLabel.text = $"{className}: {detection.confidence:P1}";
                    }
                }

                activeDetectionBoxes.Add(boxObj);
            }
        }

        private void ClearDetectionBoxes()
        {
            foreach (var box in activeDetectionBoxes)
            {
                if (box != null)
                    DestroyImmediate(box);
            }
            activeDetectionBoxes.Clear();
        }

        private void UpdatePerformanceMetrics(float frameTime)
        {
            frameTimeHistory.Enqueue(frameTime);
            if (frameTimeHistory.Count > MAX_FRAME_HISTORY)
                frameTimeHistory.Dequeue();

            if (fpsText != null && frameTimeHistory.Count > 0)
            {
                float avgFrameTime = 0f;
                foreach (float time in frameTimeHistory)
                    avgFrameTime += time;
                avgFrameTime /= frameTimeHistory.Count;

                float avgFPS = 1f / avgFrameTime;
                fpsText.text = $"FPS: {avgFPS:F1}";
            }
        }

        private void OnConfidenceChanged(float value)
        {
            if (confidenceText != null)
                confidenceText.text = $"Confidence: {value:P1}";

            if (detector != null && detector.IsReady)
            {
                detector.UpdateConfiguration(newConfidenceThreshold: value);
            }
        }

        private void UpdateStatusText(string status)
        {
            if (statusText != null)
                statusText.text = status;
        }

        void OnDestroy()
        {
            StopDetection();
            
            if (detector != null)
            {
                detector.OnDetectionComplete -= OnDetectionComplete;
                detector.OnError -= OnDetectionError;
            }

            if (captureTexture != null)
                DestroyImmediate(captureTexture);

            if (renderTexture != null)
            {
                renderTexture.Release();
                DestroyImmediate(renderTexture);
            }
        }

        void OnApplicationPause(bool pauseStatus)
        {
            if (pauseStatus && isDetecting)
            {
                StopDetection();
            }
        }

        // Public methods for external control
        public void SetConfidenceThreshold(float threshold)
        {
            if (confidenceSlider != null)
                confidenceSlider.value = threshold;
        }

        public void SetTargetFPS(int fps)
        {
            targetFPS = Mathf.Clamp(fps, 1, 30);
        }

        public bool IsDetecting => isDetecting;
        public MilitaryTargetDetector.DetectionResult LastResult { get; private set; }
    }

    /// <summary>
    /// Simple detection box UI component
    /// </summary>
    public class DetectionBox : MonoBehaviour
    {
        public Text labelText;
        public Image borderImage;
        public Image backgroundImage;

        public void SetDetection(MilitaryTargetDetector.Detection detection)
        {
            if (labelText != null)
            {
                string className = MilitaryTargetDetector.GetClassName(detection.targetClass);
                labelText.text = $"{className}\n{detection.confidence:P1}";
            }

            Color classColor = MilitaryTargetDetector.GetClassColor(detection.targetClass);
            
            if (borderImage != null)
                borderImage.color = classColor;
                
            if (backgroundImage != null)
            {
                Color bgColor = classColor;
                bgColor.a = 0.3f; // Semi-transparent background
                backgroundImage.color = bgColor;
            }
        }
    }
}