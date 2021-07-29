using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_Classification_.DataModels
{
    class ModelPrediction
    {
        [ColumnName("PredictedLabel")]
        public int Label { get; set; }
        public float[] Score { get; set; }
    }
}
