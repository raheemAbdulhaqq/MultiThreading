using Microsoft.ML;
using MLNET_Classification.DataModels;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;

namespace MLNET_Classification
{
    class Program
    {
        static readonly string _dataFilePath = Path.Combine(Environment.CurrentDirectory, "Data", "mushrooms.csv");
        static void Main(string[] args)
        {
            //Creating MLContxet model to be shared accross model building, validation and prediction process
            MLContext mlContext = new MLContext();

            //Loading the data from csv files
            //Splitting the dataset into train/test sets
            TrainTestData mushroomTrainTestData = LoadData(mlContext, testDataFraction: 0.25);

            //Creating data transformation pipeline which transforma the data a form acceptable by model
            //Returns an object of type IEstimator<ITransformer>
            var pipeline = ProcessData(mlContext);

            //passing the transformation pipeline and training dataset to crossvalidate and build the model
            //returns the model object of type ITransformer 
            var trainedModel = BuildAndTrain(mlContext, pipeline, mushroomTrainTestData.TrainSet);

            //Sample datainput for predicrtion
            var mushroomInput1 = new MushroomModelInput
            {
                skill1 = "ICU",
                skill2 = "Empathy",
                skill3 = "System Thinking"

            };

            //Sample datainput for predicrtion
            var mushroomInput2 = new MushroomModelInput
            {
                skill1 = "Home Care",
                skill2 = "ICU",
                skill3 = "Empathy"
            };

            //passing trained model and sample input data to make single prediction 
            var result = PredictSingleResult(mlContext, trainedModel, mushroomInput1);

            Console.WriteLine("================================= Single Prediction Result ===============================");
            // Evaluate(mlContext, pipeline, trainedModel,  mushroomTrainTestData.TestSet);
            Console.WriteLine($"Predicted Result: {result.Label}");

            Console.ReadKey();




        }

        public static TrainTestData LoadData(MLContext mlContext, double testDataFraction)
        {
            //Read data
            IDataView mushroomDataView = mlContext.Data.LoadFromTextFile<MushroomModelInput>(_dataFilePath, hasHeader: true, separatorChar: ',', allowSparse: false);

            TrainTestData mushroomTrainTestData = mlContext.Data.TrainTestSplit(mushroomDataView, testFraction: testDataFraction);

            return mushroomTrainTestData;
        }

        public static IEstimator<ITransformer> ProcessData(MLContext mlContext)
        {
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(MushroomModelInput.organization))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "skill1", outputColumnName: "skill1Featurized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "skill2", outputColumnName: "skill2Featurized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "skill3", outputColumnName: "skill3Featurized"))
                .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", inputColumnNames: new string[]{ "skill1Featurized", "skill2Featurized" , "skill3Featurized"}));
            return pipeline;
        }

        public static ITransformer BuildAndTrain(MLContext mlContext, IEstimator<ITransformer> pipeline, IDataView trainDataView)
        {

            //   PeekDataViewInConsole(mlContext, trainDataView, pipeline, 2);




            var trainPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features", numberOfIterations: 10)))
                                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("=============== Starting 10 fold cross validation ===============");
            var crossValResults = mlContext.MulticlassClassification.CrossValidate(data: trainDataView, estimator: trainPipeline, numberOfFolds: 10, labelColumnName: "Label");

            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();


            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();


            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();


            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();


            //Console.WriteLine($"*************************************************************************************************************");
            //Console.WriteLine($"*       Metrics Classification model      ");
            //Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            //Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###} ");
            //Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###} ");
            //Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###} ");
            //Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###} ");
            //Console.WriteLine($"*************************************************************************************************************");


            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = trainPipeline.Fit(trainDataView);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;

        }

        public static MushroomModelPrediction PredictSingleResult(MLContext mlContext, ITransformer model, MushroomModelInput input)
        {

            //Creating the prediction engine which takes data model input and output
            var predictEngine = mlContext.Model.CreatePredictionEngine<MushroomModelInput, MushroomModelPrediction>(model);

            var predOutput = predictEngine.Predict(input);

            return predOutput;


        }


        // This method using 'DebuggerExtensions.Preview()' should only be used when debugging/developing, not for release/production trainings
        //public static void PeekDataViewInConsole(MLContext mlContext, IDataView dataView, IEstimator<ITransformer> pipeline, int numberOfRows = 4)
        {
            string msg = string.Format("Peek data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());


            //https://github.com/dotnet/machinelearning/blob/master/docs/code/MlNetCookBook.md#how-do-i-look-at-the-intermediate-data
            var transformer = pipeline.Fit(dataView);
            var transformedData = transformer.Transform(dataView);

            // 'transformedData' is a 'promise' of data, lazy-loading. call Preview  
            //and iterate through the returned collection from preview.

            var preViewTransformedData = transformedData.Preview(maxRows: numberOfRows);

            foreach (var row in preViewTransformedData.RowView)
            {
                var ColumnCollection = row.Values;
                string lineToPrint = "Row--> ";
                foreach (KeyValuePair<string, object> column in ColumnCollection)
                {
                    lineToPrint += $"| {column.Key}:{column.Value}";
                }
                Console.WriteLine(lineToPrint + "\n");
            }
        }

        //     .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
    }
}
