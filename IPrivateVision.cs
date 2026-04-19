using OutSystems.ExternalLibraries.SDK;

namespace PrivateVision
{
    [OSInterface(Name = "PrivateVision", Description = "Secure, free, and fully local AI Vision for ODC.", IconResourceName = "PrivateVision.resources.privatevision_logo.png")]
    public interface IPrivateVision
    {
        [OSAction(ReturnName = "Response")]
        public Response Call(
            string UserPrompt,
            byte[] Image
        );

        [OSAction(Description = "Dummy action to be used to keep the container alive, thus preventing \"cold-starts\".")]
        public void Ping();
    }
}
