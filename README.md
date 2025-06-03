1. Clone the repo - `git clone https://github.com/vaghani-shreya/MLOps-BikeShare.git`
2. Navigate to the folder using the command `cd BikeSharingMLops`
3. Create a virtual environment
     - For Mac, use the command `source .venv/bin/activate`
     - For Windows, use the command - `.venv\Scripts\activate`
4. After activating the Virtual environment, navigate to the **notebooks** folder and run the bikeShare.ipynb
5. After running the bikeshare.ipynb, the data and information along with the models will be logged into the MLFlow UI, and the run information can be found under the notebook tab. he path is
   **\notebooks\mlruns\0**
6. Run the following command in the terminal to start MLFlow ui `mlflow ui --port 5000`. Once MLFlow is running, copy and paste `http://127.0.0.1:5000` into your
   desired browser to check that MLFlow runs under the **Run** tab.
7. If it says the port is busy, kill the processes running on the port and **retry Step 6**.
8. After the Model is registered and trained, we serve the model locally. Get the Run ID of the saved model. It should be under **\notebooks\mlruns\0** path.
9. Run the following command in your terminal: `mlflow models serve -m runs:/<Your-run-id>/bikePredModel --port 5001 --no-conda`
10. After running the command, visit the URL, `http://127.0.0.1:5001/invocations` in your desired browser.
11. It can also be tested using Curl Post command to gather predictions. Additionally, a Python script can also be used to test it.
12. To stop the server, press `CTRL + C` when the model is being served.

   



