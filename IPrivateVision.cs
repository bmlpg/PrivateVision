using OutSystems.ExternalLibraries.SDK;

namespace PrivateVision
{
    [OSInterface(Name = "PrivateVision", Description = "Free and fully local AI Vision for ODC.", IconResourceName = "PrivateVision.resources.privatevision_logo.png")]
    public interface IPrivateVision
    {
        [OSAction(ReturnName = "Response")]
        public Response Call(
            [OSParameter(Description = "Task description.")]
            string UserPrompt,
            [OSParameter(Description = "Image to process.")]
            byte[] Image,
            [OSParameter(Description = "Lower (0.0 - 0.2) for precise data extraction. Higher (0.7+) for creative summaries. Default: 0.0.")]
            float Temperature = 0.0f,
            [OSParameter(Description = "Controls the maximum length of the response. Increase for long summaries; decrease to save processing time. Default: 256.")]
            int MaxTokens = 256,
            [OSParameter(Description = "The \"memory\" window for the AI. Affects RAM usage. Min: 128. Default: 1024.")]
            int ContextSize = 1024,
            [OSParameter(Description = "Number of CPU cores to use for math operations. Ensure it is at least 1. If set higher than the host’s logical cores, performance will actually degrade. Default 1.")]
            int Threads = 1
        );

        [OSAction(Description = "Dummy action to be used to keep the container alive, thus preventing \"cold-starts\".")]
        public void Ping();
    }
}
