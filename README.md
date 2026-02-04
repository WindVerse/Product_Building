# Product Building

## Building the Final Product with Trained Models

### How to Add the Flag and Model to Unity

1. **Export the flag mesh** from Blender as `.obj`.
2. **Convert the trained PyTorch model (`.pth`)** to **ONNX (`.onnx`)** format.
3. **Convert wind data** into `.txt` files.
4. Copy the following into your **Unity Project > Assets > Flag_Simulation** folder:

   - `flag.obj`
   - `model.onnx`
   - all `wind_data.txt` to a single folder
   - `FlagSimulator.cs`
5. In Unity:

   - Drag **`flag.obj`** from *Assets* into the **Hierarchy**.
   - Right-click the object → **Prefab** → **Unpack**.
   - Select the **main object** (often named *Default*, *Flag*, or similar).
   - Add the script:
     - Drag **`FlagSimulator.cs`** onto the object’s **Inspector**.
   - Assign references:
     - Drag the **`model.onnx`** file into the `Model Asset` field.
     - Drag the **wind data `.txt`** files into the `Wind Sequence` field.
6. **Press Play** to run the simulation in Unity.
