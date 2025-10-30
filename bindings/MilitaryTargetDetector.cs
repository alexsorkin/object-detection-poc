using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace MilitaryTargetDetection
{
    /// <summary>
    /// Unity C# wrapper for the military target detection library
    /// </summary>
    public class MilitaryTargetDetector : MonoBehaviour, IDisposable
    {
        #region Native Library Imports

        private const string DLL_NAME = "military_target_detector";

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern int mtd_init();

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr mtd_version();

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern void mtd_free_string(IntPtr ptr);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern int mtd_create_detector(
            string model_path,
            int input_width,
            int input_height,
            float confidence_threshold,
            float nms_threshold,
            int max_detections,
            int use_gpu
        );

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern int mtd_destroy_detector(int detector_id);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr mtd_detect_image(
            int detector_id,
            IntPtr image_data,
            int data_length,
            int width,
            int height,
            int format
        );

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr mtd_detect_file(int detector_id, string image_path);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern void mtd_free_result(IntPtr result);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr mtd_get_class_name(int class_id);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern void mtd_get_class_color(int class_id, IntPtr rgb);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern int mtd_update_config(
            int detector_id,
            float confidence_threshold,
            float nms_threshold,
            int max_detections
        );

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern int mtd_warmup(int detector_id);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern int mtd_get_class_count();

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern void mtd_cleanup();

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr mtd_get_last_error();

        #endregion

        #region Native Structures

        [StructLayout(LayoutKind.Sequential)]
        private struct CDetection
        {
            public int class_id;
            public float confidence;
            public float x;
            public float y;
            public float width;
            public float height;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct CDetectionResult
        {
            public IntPtr detections;
            public int count;
            public float inference_time_ms;
            public int image_width;
            public int image_height;
        }

        #endregion

        #region Public Types

        /// <summary>
        /// Target class enumeration
        /// </summary>
        public enum TargetClass
        {
            ArmedPersonnel = 0,
            RocketLauncher = 1,
            MilitaryVehicle = 2,
            HeavyWeapon = 3
        }

        /// <summary>
        /// Image format enumeration
        /// </summary>
        public enum ImageFormat
        {
            RGB = 0,
            BGR = 1,
            RGBA = 2,
            BGRA = 3,
            Grayscale = 4
        }

        /// <summary>
        /// Bounding box structure
        /// </summary>
        [Serializable]
        public struct BoundingBox
        {
            public float x;
            public float y;
            public float width;
            public float height;

            public BoundingBox(float x, float y, float width, float height)
            {
                this.x = x;
                this.y = y;
                this.width = width;
                this.height = height;
            }

            public Vector2 Center => new Vector2(x + width / 2f, y + height / 2f);
            public float Area => width * height;

            public Rect ToRect(int imageWidth, int imageHeight)
            {
                return new Rect(
                    x * imageWidth,
                    y * imageHeight,
                    width * imageWidth,
                    height * imageHeight
                );
            }
        }

        /// <summary>
        /// Detection result structure
        /// </summary>
        [Serializable]
        public struct Detection
        {
            public TargetClass targetClass;
            public float confidence;
            public BoundingBox boundingBox;

            public Detection(TargetClass targetClass, float confidence, BoundingBox boundingBox)
            {
                this.targetClass = targetClass;
                this.confidence = confidence;
                this.boundingBox = boundingBox;
            }
        }

        /// <summary>
        /// Detection result collection
        /// </summary>
        [Serializable]
        public class DetectionResult
        {
            public Detection[] detections;
            public float inferenceTimeMs;
            public int imageWidth;
            public int imageHeight;
            public DateTime timestamp;

            public int Count => detections?.Length ?? 0;

            public DetectionResult()
            {
                detections = new Detection[0];
                timestamp = DateTime.Now;
            }
        }

        #endregion

        #region Configuration

        [Header("Model Configuration")]
        [SerializeField] private string modelPath = "Models/military_targets.onnx";
        [SerializeField] private Vector2Int inputSize = new Vector2Int(640, 640);
        [SerializeField] private bool useGPU = true;

        [Header("Detection Parameters")]
        [Range(0f, 1f)]
        [SerializeField] private float confidenceThreshold = 0.5f;
        [Range(0f, 1f)]
        [SerializeField] private float nmsThreshold = 0.45f;
        [SerializeField] private int maxDetections = 100;

        [Header("Performance")]
        [SerializeField] private bool warmupOnStart = true;
        [SerializeField] private bool enableLogging = true;

        #endregion

        #region Events

        public event Action<DetectionResult> OnDetectionComplete;
        public event Action<string> OnError;

        #endregion

        #region Private Fields

        private int detectorId = -1;
        private bool isInitialized = false;
        private static bool libraryInitialized = false;

        #endregion

        #region Unity Lifecycle

        void Start()
        {
            InitializeDetector();
        }

        void OnDestroy()
        {
            Dispose();
        }

        void OnApplicationPause(bool pauseStatus)
        {
            if (pauseStatus)
            {
                // Pause detection processing if needed
            }
            else
            {
                // Resume detection processing
            }
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Initialize the detector with current configuration
        /// </summary>
        public bool InitializeDetector()
        {
            try
            {
                // Initialize library once
                if (!libraryInitialized)
                {
                    int initResult = mtd_init();
                    if (initResult != 0)
                    {
                        LogError("Failed to initialize detection library");
                        return false;
                    }
                    libraryInitialized = true;
                    Log("Detection library initialized successfully");
                }

                // Create detector instance
                string fullModelPath = System.IO.Path.Combine(Application.streamingAssetsPath, modelPath);
                
                detectorId = mtd_create_detector(
                    fullModelPath,
                    inputSize.x,
                    inputSize.y,
                    confidenceThreshold,
                    nmsThreshold,
                    maxDetections,
                    useGPU ? 1 : 0
                );

                if (detectorId < 0)
                {
                    LogError($"Failed to create detector. Model path: {fullModelPath}");
                    return false;
                }

                isInitialized = true;
                Log($"Detector created successfully (ID: {detectorId})");

                // Warm up if enabled
                if (warmupOnStart)
                {
                    WarmUp();
                }

                return true;
            }
            catch (Exception ex)
            {
                LogError($"Exception during detector initialization: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Detect targets in a Texture2D
        /// </summary>
        public DetectionResult DetectInTexture(Texture2D texture)
        {
            if (!IsValidForDetection(texture))
                return new DetectionResult();

            try
            {
                // Convert texture to byte array
                Color32[] pixels = texture.GetPixels32();
                byte[] imageData = new byte[pixels.Length * 3]; // RGB format

                // Convert Color32 to RGB bytes
                for (int i = 0; i < pixels.Length; i++)
                {
                    imageData[i * 3] = pixels[i].r;
                    imageData[i * 3 + 1] = pixels[i].g;
                    imageData[i * 3 + 2] = pixels[i].b;
                }

                return DetectInImageData(imageData, texture.width, texture.height, ImageFormat.RGB);
            }
            catch (Exception ex)
            {
                LogError($"Error detecting in texture: {ex.Message}");
                return new DetectionResult();
            }
        }

        /// <summary>
        /// Detect targets in raw image data
        /// </summary>
        public DetectionResult DetectInImageData(byte[] imageData, int width, int height, ImageFormat format)
        {
            if (!isInitialized || imageData == null || imageData.Length == 0)
            {
                LogError("Detector not initialized or invalid image data");
                return new DetectionResult();
            }

            try
            {
                // Pin memory for native call
                GCHandle handle = GCHandle.Alloc(imageData, GCHandleType.Pinned);
                IntPtr dataPtr = handle.AddrOfPinnedObject();

                IntPtr resultPtr = mtd_detect_image(
                    detectorId,
                    dataPtr,
                    imageData.Length,
                    width,
                    height,
                    (int)format
                );

                handle.Free();

                if (resultPtr == IntPtr.Zero)
                {
                    LogError("Detection failed - null result");
                    return new DetectionResult();
                }

                // Convert native result to managed
                DetectionResult result = ConvertNativeResult(resultPtr);
                mtd_free_result(resultPtr);

                OnDetectionComplete?.Invoke(result);
                return result;
            }
            catch (Exception ex)
            {
                LogError($"Error during detection: {ex.Message}");
                return new DetectionResult();
            }
        }

        /// <summary>
        /// Detect targets from image file
        /// </summary>
        public DetectionResult DetectInFile(string imagePath)
        {
            if (!isInitialized || string.IsNullOrEmpty(imagePath))
            {
                LogError("Detector not initialized or invalid image path");
                return new DetectionResult();
            }

            try
            {
                IntPtr resultPtr = mtd_detect_file(detectorId, imagePath);

                if (resultPtr == IntPtr.Zero)
                {
                    LogError($"Detection failed for file: {imagePath}");
                    return new DetectionResult();
                }

                DetectionResult result = ConvertNativeResult(resultPtr);
                mtd_free_result(resultPtr);

                OnDetectionComplete?.Invoke(result);
                return result;
            }
            catch (Exception ex)
            {
                LogError($"Error detecting in file: {ex.Message}");
                return new DetectionResult();
            }
        }

        /// <summary>
        /// Update detector configuration
        /// </summary>
        public bool UpdateConfiguration(float? newConfidenceThreshold = null, 
                                      float? newNmsThreshold = null, 
                                      int? newMaxDetections = null)
        {
            if (!isInitialized)
            {
                LogError("Detector not initialized");
                return false;
            }

            float confThresh = newConfidenceThreshold ?? confidenceThreshold;
            float nmsThresh = newNmsThreshold ?? nmsThreshold;
            int maxDet = newMaxDetections ?? maxDetections;

            int result = mtd_update_config(detectorId, confThresh, nmsThresh, maxDet);
            
            if (result == 0)
            {
                confidenceThreshold = confThresh;
                nmsThreshold = nmsThresh;
                maxDetections = maxDet;
                Log("Configuration updated successfully");
                return true;
            }
            else
            {
                LogError("Failed to update configuration");
                return false;
            }
        }

        /// <summary>
        /// Warm up the detector (run dummy inference)
        /// </summary>
        public bool WarmUp()
        {
            if (!isInitialized)
            {
                LogError("Detector not initialized");
                return false;
            }

            int result = mtd_warmup(detectorId);
            if (result == 0)
            {
                Log("Detector warmed up successfully");
                return true;
            }
            else
            {
                LogError("Failed to warm up detector");
                return false;
            }
        }

        /// <summary>
        /// Get class name for target class
        /// </summary>
        public static string GetClassName(TargetClass targetClass)
        {
            IntPtr namePtr = mtd_get_class_name((int)targetClass);
            if (namePtr != IntPtr.Zero)
            {
                string name = Marshal.PtrToStringAnsi(namePtr);
                mtd_free_string(namePtr);
                return name ?? targetClass.ToString();
            }
            return targetClass.ToString();
        }

        /// <summary>
        /// Get class color for target class
        /// </summary>
        public static Color GetClassColor(TargetClass targetClass)
        {
            byte[] rgb = new byte[3];
            GCHandle handle = GCHandle.Alloc(rgb, GCHandleType.Pinned);
            
            mtd_get_class_color((int)targetClass, handle.AddrOfPinnedObject());
            handle.Free();

            return new Color(rgb[0] / 255f, rgb[1] / 255f, rgb[2] / 255f, 1f);
        }

        /// <summary>
        /// Get number of available target classes
        /// </summary>
        public static int GetClassCount()
        {
            return mtd_get_class_count();
        }

        /// <summary>
        /// Check if detector is ready for detection
        /// </summary>
        public bool IsReady => isInitialized && detectorId >= 0;

        #endregion

        #region Private Methods

        private bool IsValidForDetection(Texture2D texture)
        {
            if (!isInitialized)
            {
                LogError("Detector not initialized");
                return false;
            }

            if (texture == null)
            {
                LogError("Texture is null");
                return false;
            }

            if (!texture.isReadable)
            {
                LogError("Texture is not readable. Enable 'Read/Write' in texture import settings.");
                return false;
            }

            return true;
        }

        private DetectionResult ConvertNativeResult(IntPtr resultPtr)
        {
            CDetectionResult nativeResult = Marshal.PtrToStructure<CDetectionResult>(resultPtr);
            
            DetectionResult result = new DetectionResult
            {
                inferenceTimeMs = nativeResult.inference_time_ms,
                imageWidth = nativeResult.image_width,
                imageHeight = nativeResult.image_height,
                detections = new Detection[nativeResult.count]
            };

            if (nativeResult.count > 0 && nativeResult.detections != IntPtr.Zero)
            {
                int structSize = Marshal.SizeOf<CDetection>();
                
                for (int i = 0; i < nativeResult.count; i++)
                {
                    IntPtr detectionPtr = new IntPtr(nativeResult.detections.ToInt64() + i * structSize);
                    CDetection nativeDetection = Marshal.PtrToStructure<CDetection>(detectionPtr);

                    result.detections[i] = new Detection(
                        (TargetClass)nativeDetection.class_id,
                        nativeDetection.confidence,
                        new BoundingBox(
                            nativeDetection.x,
                            nativeDetection.y,
                            nativeDetection.width,
                            nativeDetection.height
                        )
                    );
                }
            }

            return result;
        }

        private void Log(string message)
        {
            if (enableLogging)
            {
                Debug.Log($"[MilitaryTargetDetector] {message}");
            }
        }

        private void LogError(string message)
        {
            Debug.LogError($"[MilitaryTargetDetector] {message}");
            OnError?.Invoke(message);
        }

        #endregion

        #region IDisposable

        public void Dispose()
        {
            if (isInitialized && detectorId >= 0)
            {
                mtd_destroy_detector(detectorId);
                detectorId = -1;
                isInitialized = false;
                Log("Detector disposed");
            }
        }

        #endregion

        #region Static Cleanup

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void StaticCleanup()
        {
            if (libraryInitialized)
            {
                mtd_cleanup();
                libraryInitialized = false;
            }
        }

        #endregion
    }

    /// <summary>
    /// Utility class for detection visualization
    /// </summary>
    public static class DetectionVisualizer
    {
        /// <summary>
        /// Draw detection bounding boxes on a texture
        /// </summary>
        public static Texture2D DrawDetections(Texture2D originalTexture, 
                                             MilitaryTargetDetector.DetectionResult result,
                                             int lineWidth = 2)
        {
            if (originalTexture == null || result == null || result.Count == 0)
                return originalTexture;

            Texture2D annotatedTexture = new Texture2D(originalTexture.width, originalTexture.height);
            annotatedTexture.SetPixels(originalTexture.GetPixels());

            foreach (var detection in result.detections)
            {
                Color classColor = MilitaryTargetDetector.GetClassColor(detection.targetClass);
                Rect bbox = detection.boundingBox.ToRect(originalTexture.width, originalTexture.height);
                
                DrawRect(annotatedTexture, bbox, classColor, lineWidth);
            }

            annotatedTexture.Apply();
            return annotatedTexture;
        }

        private static void DrawRect(Texture2D texture, Rect rect, Color color, int lineWidth)
        {
            int x1 = Mathf.Clamp((int)rect.x, 0, texture.width - 1);
            int y1 = Mathf.Clamp((int)rect.y, 0, texture.height - 1);
            int x2 = Mathf.Clamp((int)(rect.x + rect.width), 0, texture.width - 1);
            int y2 = Mathf.Clamp((int)(rect.y + rect.height), 0, texture.height - 1);

            // Draw horizontal lines
            for (int x = x1; x <= x2; x++)
            {
                for (int i = 0; i < lineWidth; i++)
                {
                    if (y1 + i < texture.height) texture.SetPixel(x, y1 + i, color);
                    if (y2 - i >= 0) texture.SetPixel(x, y2 - i, color);
                }
            }

            // Draw vertical lines
            for (int y = y1; y <= y2; y++)
            {
                for (int i = 0; i < lineWidth; i++)
                {
                    if (x1 + i < texture.width) texture.SetPixel(x1 + i, y, color);
                    if (x2 - i >= 0) texture.SetPixel(x2 - i, y, color);
                }
            }
        }
    }
}