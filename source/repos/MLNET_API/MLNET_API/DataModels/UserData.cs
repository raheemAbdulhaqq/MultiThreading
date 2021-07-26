using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLNET_API.DataModels
{
    public class UserData
    {
        [LoadColumn(0)]
        public string skill1 { get; set; }
        [LoadColumn(1)]
        public string skill2 { get; set; }
        [LoadColumn(2)]
        public string skill3 { get; set; }
        [LoadColumn(3)]
        //[ColumnName("Label")]
        public string organization { get; set; }
    }
}
