## Installing FFmpeg

This project requires **FFmpeg** to save or render movie simulations.  
Please install FFmpeg according to your operating system:

---

### Windows
1. Download FFmpeg from the [official website](https://ffmpeg.org/download.html).
   - Under "Windows", choose a build from sites like Gyan.dev or BtbN.
2. Extract the downloaded `.zip` file.
3. Add the `bin` folder (inside the extracted folder) to your **System PATH**:
   - Search for "Environment Variables" → Edit the **PATH** variable → Add the path to the `bin` folder.
4. Open a new Command Prompt and verify the installation:
   ```bash
   ffmpeg -version
   ```

### macO
If you have Homebrew installed, run:
```bash
   brew install ffmpeg
```
Then verify: 
```bash
ffmpeg -version
```
If you don't have Homebrew installed, you can install it first by running:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Linux (Ubuntu/Debian)
Update your package list and install FFmpeg:
```bash
sudo apt update
sudo apt install ffmpeg
```
Then verify:
```bash
ffmpeg -version
```


