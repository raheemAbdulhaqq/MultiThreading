using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNET_Classification.DataModels
{
    class MushroomModelPrediction
    {

        [ColumnName("PredictedLabel")]
        public string Label { get; set; }
        public float[] Score { get; set; }
    }
}
