using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLNET_API.DataModels
{
    public class UserPrediction : UserData
    {
        [ColumnName("PredictedLabel")]
        public string Label { get; set; }
        public float[] Score { get; set; }
    }
}
