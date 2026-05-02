using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using Microsoft.Extensions.Logging;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace PrivateVision
{
    public class PrivateVision : IPrivateVision
    {
        private readonly ILogger _logger;

        private static bool initialized = false;

        private static LLamaWeights? model;
        private static MtmdWeights? clipModel;

        private static int currentContextSize = 0;
        private static int currentThreads = 0;

        public PrivateVision(ILogger logger)
        {
            _logger = logger;

            if(initialized) return;

            LLama.Native.NativeLibraryConfig.All.WithLogCallback((level, message) =>
            {
                bool isError = level == LLama.Native.LLamaLogLevel.Error;
                bool isWarning = level == LLama.Native.LLamaLogLevel.Warning;

                if (!isError && !isWarning) return;

                using var activity = Activity.Current?.Source.StartActivity("PrivateVision.NativeLog");

                if (isError)
                {
                    _logger.LogError($"LL# NATIVE ERROR: {message}");
                    activity?.SetStatus(ActivityStatusCode.Error, message);
                }
                else
                {
                    _logger.LogWarning($"LL# NATIVE WARNING: {message}");
                }
            });

            initialized = true;
        }

        [DllImport("libc")]
        public static extern int malloc_trim(uint pad);

        public Response Call(
            string UserPrompt,
            byte[] Image,
            float Temperature = 0.0f,
            int MaxTokens = 256,
            int ContextSize = 1024,
            int Threads = 1
        )
        {

            if (ContextSize < 128)
            {
                throw new ArgumentException("ContextSize must be greater or equal than 128");
            }

            if (Threads < 1)
            {
                throw new ArgumentException("Threads must be greater or equal than 1");
            }

            if (Temperature < 0.0f || Temperature > 2.0f)
            {
                throw new ArgumentException("Temperature must be between 0.0 and 2.0");
            }

            if (MaxTokens < 128)
            {
                throw new ArgumentException("MaxTokens must be greated or equal than 128");
            }

            Response response = new Response();

            try
            {
                var modelPath = Path.Combine(Path.GetTempPath(), "model.gguf");
                var parameters = new ModelParams(modelPath)
                {
                    ContextSize = (uint)ContextSize,
                    GpuLayerCount = 0,
                    UseMemorymap = false,
                    Threads = Threads
                };

                var mtmdParameters = MtmdContextParams.Default();
                mtmdParameters.UseGpu = false;

                if (model is null)
                {
                    Task.Run(async () => await DownloadModels()).GetAwaiter().GetResult();
                    model = LLamaWeights.LoadFromFile(parameters);
                    var mmProjPath = Path.Combine(Path.GetTempPath(), "mmProj.gguf");
                    clipModel = MtmdWeights.LoadFromFile(mmProjPath, model, mtmdParameters);
                }
                else if (currentContextSize != ContextSize || currentThreads != Threads)
                {
                    model.Dispose();
                    clipModel.Dispose();
                    model = LLamaWeights.LoadFromFile(parameters);
                    var mmProjPath = Path.Combine(Path.GetTempPath(), "mmProj.gguf");
                    clipModel = MtmdWeights.LoadFromFile(mmProjPath, model, mtmdParameters);
                    currentContextSize = ContextSize;
                    currentThreads = Threads;
                }
                
                using var context = model.CreateContext(parameters);

                byte[] resizedImage = ResizeImageForAI(Image, 384);
                using var image = clipModel.LoadMedia(resizedImage);

                // 4. Create the Multimodal Prompt (ChatML format)
                string fullPrompt = $"<|im_start|>user\n<image>\n{UserPrompt}<|im_end|>\n<|im_start|>assistant\n";

                // 5. Inference
                var executor = new InteractiveExecutor(context, clipModel);
                executor.Embeds.Add(image);

                using var pipeline = new DefaultSamplingPipeline()
                {
                    Temperature = Temperature,
                    RepeatPenalty = 1.1f,
                    PresencePenalty = 0.1f
                };

                // We use the clipModel to 'project' the image into the context
                var inferenceParams = new InferenceParams()
                {
                    MaxTokens = MaxTokens, 
                    AntiPrompts = new[] { "<|im_end|>" },
                    SamplingPipeline = pipeline
                };

                Stopwatch sw = Stopwatch.StartNew();

                var responseBuilder = new System.Text.StringBuilder();

                // This is the magic line where vision meets text
                Task.Run(async () =>
                {
                    await foreach (var token in executor.InferAsync(fullPrompt, inferenceParams))
                    {
                        responseBuilder.Append(token);
                    }
                }).GetAwaiter().GetResult();

                response.Result = GetJsonOrOriginal(responseBuilder.ToString().Trim());

                sw.Stop();
                response.Duration = sw.ElapsedMilliseconds;

                var process = Process.GetCurrentProcess();
                process.Refresh();
                response.TotalMemoryMB = (int)(process.WorkingSet64 / (1024 * 1024));
            }
            finally
            {
                try
                {
                    malloc_trim(0);
                }
                catch
                {
                    /* Fallback for non-linux environments */
                }
            }
            
            return response;
            // Implementation here
        }

        public void Ping()
        {
        }

        private async Task DownloadModels()
        {
            var modelDir = Path.GetTempPath();
            var tasks = new List<Task>();

            var filesToDownload = new[]
            {
                (Url: "https://huggingface.co/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/resolve/main/SmolVLM2-256M-Video-Instruct-Q8_0.gguf", Name: "model.gguf"),
                (Url: "https://huggingface.co/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf", Name: "mmProj.gguf")
            };

            using var client = new HttpClient();
            client.DefaultRequestHeaders.Add("User-Agent", "OutSystems-PrivateVision-Plugin-ODC");

            foreach (var file in filesToDownload)
            {
                string finalPath = Path.Combine(modelDir, file.Name);
                if (!File.Exists(finalPath))
                {
                    tasks.Add(DownloadFileAsync(client, file.Url, finalPath));
                }
            }

            if (tasks.Count > 0)
            {
                _logger.LogInformation($"Downloading {tasks.Count} model files in parallel...");
                await Task.WhenAll(tasks);
            }
        }

        private async Task DownloadFileAsync(HttpClient client, string url, string finalPath)
        {
            string tempPath = finalPath + ".tmp";

            // Cleanup any failed previous attempts
            if (File.Exists(tempPath)) File.Delete(tempPath);

            try
            {
                using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
                response.EnsureSuccessStatusCode();

                using (var networkStream = await response.Content.ReadAsStreamAsync())
                using (var fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    await networkStream.CopyToAsync(fileStream);
                    await fileStream.FlushAsync();
                }

                // Final atomic move
                if (File.Exists(finalPath)) File.Delete(tempPath);
                else File.Move(tempPath, finalPath);

                _logger.LogInformation($"Finished: {Path.GetFileName(finalPath)}");
            }
            catch (Exception ex)
            {
                if (File.Exists(tempPath)) File.Delete(tempPath);
                _logger.LogError($"Download failed for {url}: {ex.Message}");
                throw;
            }
        }

        private static byte[] ResizeImageForAI(byte[] imageBuffer, int maxDimension)
        {
            using (MemoryStream inStream = new MemoryStream(imageBuffer))
            using (MemoryStream outStream = new MemoryStream())
            {
                // 1. Load the image from the buffer
                using (Image image = Image.Load(inStream))
                {
                    // 2. Resize using the 'Lanczos3' sampler (best quality for OCR/Text)
                    image.Mutate(x => x.Resize(new ResizeOptions
                    {
                        Size = new Size(maxDimension, maxDimension),
                        Mode = ResizeMode.Max // Maintains aspect ratio within the bounds
                    }));

                    // 3. Save as JPEG or PNG (JPEG is usually smaller for RAM)
                    image.SaveAsJpeg(outStream);
                }

                return outStream.ToArray();
            }
        }

        public string GetJsonOrOriginal(string input)
        {
            if (string.IsNullOrWhiteSpace(input)) return String.Empty;

            // This regex looks for the first '{' and the last '}' and everything in between.
            // RegexOptions.Singleline is CRITICAL because it allows the '.' to match newlines.
            var match = Regex.Match(input, @"\{.*\}", RegexOptions.Singleline);

            if (match.Success)
            {
                return match.Value;
            }

            // Fallback if no braces are found
            return input;
        }
    }
}
