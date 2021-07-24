using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNET_Classification.DataModels
{
    class MushroomModelInput
    {
        [LoadColumn(0)]
        public string skill1 { get; set; }
        [LoadColumn(1)]
        public string skill2 { get; set; }
        [LoadColumn(2)]
        public string skill3 { get; set; }
        [LoadColumn(3)]
        public string organization { get; set; }

    }
}
