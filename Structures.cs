using OutSystems.ExternalLibraries.SDK;

namespace PrivateVision
{
    [OSStructure(Description = "")]
    public struct Response
    {
        [OSStructureField(DataType = OSDataType.Text, IsMandatory = false)]
        public string Result;
        [OSStructureField(DataType = OSDataType.LongInteger, Description = "Duration in milliseconds.", IsMandatory = false)]
        public long Duration;
    }
}
