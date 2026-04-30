using OutSystems.ExternalLibraries.SDK;

namespace PrivateVision
{
    [OSStructure(Description = "")]
    public struct Response
    {
        [OSStructureField(DataType = OSDataType.Text, Description = "Result in text.", IsMandatory = false)]
        public string Result;
        [OSStructureField(DataType = OSDataType.LongInteger, Description = "Duration in milliseconds.", IsMandatory = false)]
        public long Duration;
        [OSStructureField(DataType = OSDataType.Integer, Description = "Total memory consumption in MB.", IsMandatory = false)]
        public int TotalMemoryMB;
    }
}
