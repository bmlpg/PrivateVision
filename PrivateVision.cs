using System.Diagnostics;
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

        public PrivateVision(ILogger logger)
        {
            _logger = logger;

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
        }

        public Response Call(string UserPrompt, byte[] Image)
        {
            Stopwatch sw = Stopwatch.StartNew();

            Task.Run(async () => await DownloadModels()).GetAwaiter().GetResult();

            Response response = new Response();

            try
            {
                var modelPath = Path.Combine(Path.GetTempPath(), "model.gguf");
                var parameters = new ModelParams(modelPath)
                {
                    ContextSize = 512,
                    GpuLayerCount = 0,
                    UseMemorymap = false,    
                    Threads = 1              
                };

                var mtmdParameters = MtmdContextParams.Default();
                mtmdParameters.UseGpu = false;

                using var model = LLamaWeights.LoadFromFile(parameters);
                using var context = model.CreateContext(parameters);

                var mmProjPath = Path.Combine(Path.GetTempPath(), "mmProj.gguf");
                using var clipModel = MtmdWeights.LoadFromFile(mmProjPath, model, mtmdParameters);

                byte[] resizedImage = ResizeImageForAI(Image, 384, 384);
                using var image = clipModel.LoadMedia(resizedImage);

                // 4. Create the Multimodal Prompt (ChatML format)
                string fullPrompt = $"<|im_start|>user\n<image>\n{UserPrompt}<|im_end|>\n<|im_start|>assistant\n";

                // 5. Inference
                var executor = new InteractiveExecutor(context, clipModel);
                executor.Embeds.Add(image);

                // We use the clipModel to 'project' the image into the context
                var inferenceParams = new InferenceParams()
                {
                    MaxTokens = 512, 
                    AntiPrompts = new[] { "<|im_end|>" },
                    SamplingPipeline = new DefaultSamplingPipeline()
                    {
                        Temperature = 0.0f,
                        RepeatPenalty = 1.1f,
                        PresencePenalty = 0.1f
                    }
                };

                // This is the magic line where vision meets text
                response.Result = Task.Run(async () =>
                {
                    var responseBuilder = new System.Text.StringBuilder();

                    await foreach (var token in executor.InferAsync(fullPrompt, inferenceParams))
                    {
                        responseBuilder.Append(token);
                    }

                    return GetJsonOrOriginal(responseBuilder.ToString().Trim());
                }).GetAwaiter().GetResult();

            }
            finally
            {
                sw.Stop();
                response.Duration = sw.ElapsedMilliseconds;

                // 6. Manual Cleanup (Critical for ODC stability)
                GC.Collect();
                GC.WaitForPendingFinalizers();
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

        private static byte[] ResizeImageForAI(byte[] imageBuffer, int width, int height)
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
                        Size = new Size(width, height),
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
