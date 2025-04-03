## Recipes
1. Install requirements via micromamba
```bash
micromamba create -y -f ./environment.yaml
```

2. Activate the environment
source a shell script depending on your shell
```bash
source setup.sh
# or
source setup.fish
```

3. Do sanity-check
```bash
./train.py
```

4. Open Aim UI
```bash
aim up --port <PORT>
```
If you are working on a remote server, you need to set up port forwarding
