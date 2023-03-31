using System;
using System.Linq;
using System.CommandLine;
using BinaryClassificationApp.Model;

namespace BinaryClassificationApp
{
    /// <summary>
    /// Class that implements the console application.
    /// </summary>
    public class Program
    {
        #region Fields

        /// <summary>
        /// Field that determines the default input file used to train the model.
        /// </summary>
        private static readonly string _trainingFileInputPath = "InputData/yelp_labelled.txt";

        /// <summary>
        /// Field that contains the model instance.
        /// </summary>
        private static ReviewSentimentModel _model = new ReviewSentimentModel("Model.mlnet");

        #endregion

        public static async Task<int> Main(string[] args)
        {
            // Command line parser
            var root = new RootCommand("ML-NET BINARY CLASSIFICATION EXAMPLE");
            var trainOption = new Option<bool>("--train", "Train the model.");
            
            root.AddOption(trainOption);
            root.SetHandler((train) =>
                {
                    if (train)
                    {
                        TrainModel();
                    }
                    else
                    {
                        ExecutePrediction();
                    }
                },
                trainOption);

            return await root.InvokeAsync(args);
        }

        /// <summary>
        /// Method that trains the ML-NET model.
        /// </summary>
        private static void TrainModel()
        {
            Console.WriteLine("Training model...");
            _model.Train(_trainingFileInputPath, '\t', false);
        }

        /// <summary>
        /// Method that executes a prediction asking the user the input comment.
        /// </summary>
        private static void ExecutePrediction()
        {
            Console.Write("Input comment: ");
            var comment = Console.ReadLine();

            if (string.IsNullOrEmpty(comment))
            {
                throw new Exception("Invalid comment.");
            }

            // TODO
            throw new NotImplementedException();
        }
    }
}
