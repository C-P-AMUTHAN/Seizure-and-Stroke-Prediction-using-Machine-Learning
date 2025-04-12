// This function dynamically generates the form fields based on the selected model type
function updateForm() {
    const modelType = document.getElementById("model_type").value;
    const form = document.getElementById("input_form");

    form.innerHTML = ''; // Clear the form

    let fields = [];

    if (modelType === "seizure") {
        fields = [
            { name: "EEG1", label: "EEG1 (float)" },
            { name: "EEG2", label: "EEG2 (float)" },
            { name: "EEG3", label: "EEG3 (float)" },
            { name: "EEG4", label: "EEG4 (float)" },
            { name: "Patient_Age", label: "Patient Age (float)" },
            { name: "Medication_Use", label: "Medication Use (0/1)" },
            { name: "phone_number", label: "Phone Number" }
        ];
    } else if (modelType === "stroke") {
        fields = [
            { name: "gender", label: "Gender (male/female)" },
            { name: "age", label: "Age (float)" },
            { name: "hypertension", label: "Hypertension (0/1)" },
            { name: "heart_disease", label: "Heart Disease (0/1)" },
            { name: "ever_married", label: "Ever Married (yes/no)" },
            { name: "work_type", label: "Work Type (private/self-employed)" },
            { name: "Residence_type", label: "Residence Type (urban/rural)" },
            { name: "avg_glucose_level", label: "Average Glucose Level (float)" },
            { name: "bmi", label: "BMI (float)" },
            { name: "smoking_status", label: "Smoking Status (never smoked/formerly smoked)" },
            { name: "phone_number", label: "Phone Number" }
        ];
    }

    // Generate form fields
    fields.forEach(field => {
        const inputDiv = document.createElement("div");
        const label = document.createElement("label");
        label.textContent = field.label;
        const input = document.createElement("input");
        input.type = "text";
        input.name = field.name;
        input.placeholder = `Enter ${field.label}`;
        inputDiv.appendChild(label);
        inputDiv.appendChild(input);
        form.appendChild(inputDiv);
    });
}

// This function sends the form data to the Flask backend for prediction
function predict() {
    const formData = {};
    const formElements = document.getElementById("input_form").elements;
    for (let i = 0; i < formElements.length; i++) {
        if (formElements[i].type !== "submit") {
            formData[formElements[i].name] = formElements[i].value;
        }
    }

    const modelType = document.getElementById("model_type").value;
    formData.model_type = modelType;

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("result").textContent = "";
            document.getElementById("error").textContent = `Error: ${data.error}`;
        } else {
            document.getElementById("error").textContent = "";
            document.getElementById("result").textContent = `Prediction: ${data.seizure_prediction}, Probability: ${data.seizure_probability}`;
        }
    })
    .catch(error => {
        document.getElementById("result").textContent = "";
        document.getElementById("error").textContent = `Error: ${error}`;
    });
}

// Initialize the form with seizure fields by default
updateForm();