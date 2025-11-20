# Create G1_EDU mitigation file
@"
From_Layer,Physical,Sensor,Data,Middleware,Decision,Application,Social_Interface
Physical,0.0,0.4,0.45,0.55,0.6,0.45,0.0
Sensor,0.35,0.0,0.35,0.3,0.45,0.5,0.55
Data,0.45,0.35,0.0,0.3,0.35,0.5,0.0
Middleware,0.4,0.3,0.3,0.0,0.5,0.4,0.5
Decision,0.45,0.45,0.35,0.45,0.0,0.35,0.3
Application,0.35,0.55,0.45,0.35,0.45,0.0,0.35
Social_Interface,0.0,0.5,0.0,0.45,0.45,0.35,0.0
"@ | Out-File -FilePath "data/G1_EDU_mitigation.csv" -Encoding utf8

# Create Pepper mitigation file
@"
From_Layer,Physical,Sensor,Data,Middleware,Decision,Application,Social_Interface
Physical,0.0,0.25,0.3,0.35,0.4,0.3,0.0
Sensor,0.3,0.0,0.2,0.15,0.3,0.4,0.5
Data,0.35,0.3,0.0,0.2,0.25,0.4,0.0
Middleware,0.2,0.15,0.2,0.0,0.3,0.25,0.4
Decision,0.35,0.35,0.3,0.35,0.0,0.25,0.2
Application,0.3,0.5,0.4,0.25,0.3,0.0,0.25
Social_Interface,0.0,0.5,0.0,0.4,0.4,0.3,0.0
"@ | Out-File -FilePath "data/Pepper_mitigation.csv" -Encoding utf8

Write-Host "✓ Created G1_EDU_mitigation.csv"
Write-Host "✓ Created Pepper_mitigation.csv"