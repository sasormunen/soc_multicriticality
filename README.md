# Model for multicritical SOC
Here you can find the code for simulating the model of coupled oscillators on an evolving network described in the paper "Self-organization to multicriticality" by Silja Sormunen, Thilo Gross and Jari Saramäki (ArXiv: https://doi.org/10.48550/arXiv.2506.04275).

The model itself is in model.py, while run_model_example.py sets the parameters and runs the model (parameters set to the default values used in the article). The model reads an initial network configuration from files/static_graphs/. Tracked parameters are written to folder "timeseries", and the network evolution is recorded to a file in the "edgechanges" folder. 
