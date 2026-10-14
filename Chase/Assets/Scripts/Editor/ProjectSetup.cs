using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace StudentTeacherAI.EditorTools
{
    /// <summary>
    /// One-time project configuration for the Web build. Also runnable headless:
    ///   Unity.exe -batchmode -quit -projectPath Chase -executeMethod StudentTeacherAI.EditorTools.ProjectSetup.Run
    /// </summary>
    public static class ProjectSetup
    {
        [MenuItem("StudentTeacherAI/Configure Project (WebGPU)")]
        public static void Run()
        {
            // WebGPU first so compute shaders are available in the browser build.
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.WebGL, false);
            PlayerSettings.SetGraphicsAPIs(BuildTarget.WebGL, new[] { GraphicsDeviceType.WebGPU });

            // Fish-Net host + bots must keep simulating when the tab loses focus.
            PlayerSettings.runInBackground = true;

            Directory.CreateDirectory(Path.Combine(Application.dataPath, "StreamingAssets"));
            AssetDatabase.Refresh();
            AssetDatabase.SaveAssets();

            var apis = string.Join(", ", PlayerSettings.GetGraphicsAPIs(BuildTarget.WebGL).Select(a => a.ToString()));
            Debug.Log($"[StudentTeacherAI] WebGL graphics APIs: {apis}; StreamingAssets ready.");
        }
    }
}
