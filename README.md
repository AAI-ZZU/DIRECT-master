# DIRECT: Decentralized Intention and Latent Rule-Emergent for Multi-Agent Cooperation under Intermittent Communication

## **experiment instructions**

### **Installation instructions**
See `requirments.txt` file for more information about how to install the dependencies.
```python
conda create -n DIRECT python=3.7.16 -y
conda activate DIRECT
pip install -r requirements.txt
```

### **Run an experiment**

You can execute the following command to run DIRECT on SMAC benchmark with a map config, such as `6h_vs_8z`:

```python
python src/main.py --config DIRECT --env_config sc2 with --env_name 6h_vs_8z --max_train_steps 5000000
```

or you can execute the following command to run DIRECT on MPE benchmark with a map config, such as `cn_3v3_classical`:
```python
python src/main.py --config DIRECT --env_config mpe with --map_name cn_3v3_classical
```

All results will be stored in the `DIRECT/results` folder. You can see the console output, config, and tensorboard logging in the `DIRECT/results/tb_logs` folder.
