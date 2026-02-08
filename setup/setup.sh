#/bash
sudo apt-get install python3-pip -y

pip3 install virtualenv

virtualenv -p python3 co2_emissions_env

source co2_emissions_env/bin/activate

pip3 install -r setup/requirements.txt
pip3 install jupyter notebook ipykernel

# Jupyter Kernel
python3 -m ipykernel install --user --name co2_emissions_env --display-name "CO2 Emissions Kernel"

echo "Setup saved successfully!"
echo "To activate the virtual environment, run: source co2_emissions_env/bin/activate"
echo "To start Jupyter Notebook, run: jupyter notebook"