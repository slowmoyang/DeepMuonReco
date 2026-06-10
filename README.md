# Muonly


## Recipes

### Install dependencies
If micromamba is not already installed on your system, you can install it easily using the following command:
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
See more details at: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

Then, create the environment using the provided `environment.yaml` file:
```bash
micromamba create -y -f ./environment.yaml
```

### Setup environment
Source a shell script depending on your shell:
- bash: `source setup.sh`
- fish: `source setup.fish`

### Training on local machine
To run a training job with a sanity-check config on your local machine, use the following command:
```bash
./train.py debug=sanity-check
```

After the sanity check passes without any errors, you can start training the model by specifying your desired configurations in the prompt.
For example:
```bash
./train.py model=latent_attention model.model_dim=128 optimizer.lr=0.0001 datamodule.batch_size=256
```

### Submit training job into a cluster
```bash
./submit.py -h
```

```bash
./submit.py --debug sanity-check
```

```bash
./submit.py --model latent_attention -a 'model.model_dim=128 optimizer.lr=0.0001 datamodule.batch_size=256'
```

### Monitor training logs with Aim UI
Open Aim UI
```bash
aim up --port <PORT>
```
If you are working on a remote server, you need to set up port forwarding
