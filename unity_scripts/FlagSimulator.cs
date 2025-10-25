using UnityEngine;
using Unity.Sentis; // Import Sentis
using System.Collections.Generic; // For Dictionaries
using System.IO; // For parsing text
using System.Linq; // For Linq operations

public class FlagSimulator : MonoBehaviour
{
    // --- Assign in Unity Inspector ---
    [Tooltip("Drag your 'flagwind_model.onnx' file here.")]
    public ModelAsset modelAsset;

    [Tooltip("Drag all your wind .txt files here. Order matters!")]
    public TextAsset[] windSequence; // Your wind .txt files

    // --- Model Internals ---
    private Model runtimeModel;
    private Worker engine; // Correct for Sentis 2.1
    
    // --- Mesh Data ---
    private Mesh flagMesh;
    private Vector3[] vertices; // Our copy of the vertices (this one gets modified)

    private Vector3[] originalVertices; // A backup of the starting mesh
    
    private Vector3 flagMean;
    private Vector3 flagStd;
    private Vector3 dispMean;
    private Vector3 dispStd;

    private readonly bool Normalize = true;
    
    // --- Simulation State ---
    private int currentFrame = 0;
    private const int NUM_VERTICES = 1024; // <-- MUST MATCH YOUR MODEL
    private const int WIND_VALUES = 24;    // 8 * 3 = 24

    void Start()
    {
        // 1. Load the model and create the inference engine
        runtimeModel = ModelLoader.Load(modelAsset);
        
        // Correct constructor for Sentis 2.1
        engine = new Worker(runtimeModel, BackendType.GPUCompute);

        // 2. Get the mesh and its vertices
        flagMesh = GetComponent<MeshFilter>().mesh;
        
        Debug.Log($"===== MY FLAG'S ACTUAL VERTEX COUNT IS: {flagMesh.vertexCount} =====");
        
        // IMPORTANT: Check vertex count
        if (flagMesh.vertexCount != NUM_VERTICES)
        {
            Debug.LogError($"Mesh vertex count ({flagMesh.vertexCount}) does not match model input ({NUM_VERTICES})!");
            this.enabled = false; // Disable script
            return;
        }
        
        // *** 2. UPDATE THIS SECTION ***
        // Get a persistent copy of the vertices
        originalVertices = flagMesh.vertices; // Save the original mesh
        vertices = new Vector3[originalVertices.Length]; // Create a new working array
        originalVertices.CopyTo(vertices, 0); // Copy the data to our working array


        // These are the calculated stats from Python model training
        flagMean = new Vector3( 0.12910223f, -0.03005729f, -0.2029736f);
        flagStd  = new Vector3( 0.2569797f,   0.25402212f,  0.22280896f);
        dispMean = new Vector3( 0.00028131f, -0.00021091f, -0.00083623f);
        dispStd  = new Vector3( 0.01928047f,  0.02270363f,  0.01108402f);
    }

    // Use FixedUpdate for physics-like simulation
    void FixedUpdate()
    {
        if (windSequence.Length == 0) return;

        // 1. GET RESULTS from the *previous* frame (if they exist)

        if (engine.PeekOutput("displacement_output") is Tensor<float> gpuDisplacements)
        {
            // 2. Copy data from GPU to a new, readable CPU tensor
            Tensor<float> displacements = gpuDisplacements.ReadbackAndClone();
            gpuDisplacements.Dispose();

            // 3. APPLY THE DISPLACEMENTS we just read
            for (int i = 0; i < NUM_VERTICES; i++)
            {
                float dx, dy, dz;

                if (Normalize)
                {
                    // *** DENORMALIZE THE OUTPUT ***
                    dx = (displacements[0, i, 0] * dispStd.x) + dispMean.x;
                    dy = (displacements[0, i, 1] * dispStd.y) + dispMean.y;
                    dz = (displacements[0, i, 2] * dispStd.z) + dispMean.z;
                }
                else
                {
                    dx = displacements[0, i, 0];
                    dy = displacements[0, i, 1];
                    dz = displacements[0, i, 2];
                }

                Vector3 displacement = new Vector3(dx, dy, dz);

                vertices[i] += displacement;
            }

            // 4. UPDATE THE ACTUAL MESH with the new vertices
            flagMesh.vertices = vertices;
            flagMesh.RecalculateNormals();

            // Clean up the CPU tensor
            displacements.Dispose();
        }

        // 5. PREPARE THE INPUTS for the *next* frame.
        Tensor<float> flagInputTensor = new Tensor<float>(new TensorShape(1, NUM_VERTICES, 3));
        Tensor<float> windInputTensor = new Tensor<float>(new TensorShape(1, 8, 3));

        // *** NORMALIZE THE INPUT ***
        for (int i = 0; i < NUM_VERTICES; i++)
        {
            if (Normalize)
            {
                // Normalize the input vertex positions
                flagInputTensor[0, i, 0] = (vertices[i].x - flagMean.x) / flagStd.x;
                flagInputTensor[0, i, 1] = (vertices[i].y - flagMean.y) / flagStd.y;
                flagInputTensor[0, i, 2] = (vertices[i].z - flagMean.z) / flagStd.z;
            }
            else
            {
                flagInputTensor[0, i, 0] = vertices[i].x;
                flagInputTensor[0, i, 1] = vertices[i].y;
                flagInputTensor[0, i, 2] = vertices[i].z;
            }
        }

        // 6. Prepare Wind Input Tensor
        LoadWindData(windSequence[currentFrame], windInputTensor); 

        // 7. Set the inputs for the new job
        engine.SetInput("flag_input", flagInputTensor);
        engine.SetInput("wind_input", windInputTensor);

        // 8. SCHEDULE THE MODEL to run on the GPU
        engine.Schedule();

        // 9. Advance to the next wind frame
        currentFrame++; // Just increment (don't loop)
        
        // 10. Clean up this frame's input tensors
        flagInputTensor.Dispose();
        windInputTensor.Dispose();
        
        // *** 3. ADD THIS CHECK ***
        // 11. Check if the simulation is finished
        if (currentFrame >= windSequence.Length)
        {
            // Reset to the beginning
            currentFrame = 0;
            
            // Reset the flag's position by copying the backup data
            originalVertices.CopyTo(vertices, 0); 
            
            // Apply the reset to the visible mesh
            flagMesh.vertices = vertices; 
            flagMesh.RecalculateNormals();
            
            Debug.Log("--- Simulation finished, resetting to initial position. ---");
        }
    }

    // Helper function to parse the .txt file into the wind tensor
    // (This function is unchanged)
    void LoadWindData(TextAsset windFile, Tensor<float> windInputTensor)
    {
        // Read all 24 float values from the text file
        string[] stringValues = windFile.text.Split(
            new[] { ' ', '\n', '\r' }, 
            System.StringSplitOptions.RemoveEmptyEntries
        );
        
        if (stringValues.Length != WIND_VALUES)
        {
            Debug.LogError($"Wind file {windFile.name} has {stringValues.Length} values, expected {WIND_VALUES}!");
            return;
        }
        
        // Parse and load into the (1, 8, 3) tensor
        int k = 0;
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                windInputTensor[0, i, j] = float.Parse(stringValues[k]);
                k++;
            }
        }
    }

    // Clean up the engine when the object is destroyed
    // (This function is unchanged)
    void OnDestroy()
    {
        engine?.Dispose();
    }
}