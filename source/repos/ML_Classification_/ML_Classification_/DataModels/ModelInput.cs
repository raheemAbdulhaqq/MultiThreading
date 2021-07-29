using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_Classification_.DataModels
{
    class ModelInput
    {
        [LoadColumn(0)]
        public string skill_first { get; set; }
        [LoadColumn(1)]
        public string skill_second { get; set; }
        [LoadColumn(2)]
        public string skill_third { get; set; }
        [LoadColumn(3)]
        public string skill_fourth { get; set; }
        [LoadColumn(4)]
        public string skill_fifth { get; set; }
        [LoadColumn(5)]
        public int id { get; set; }
    }
}
