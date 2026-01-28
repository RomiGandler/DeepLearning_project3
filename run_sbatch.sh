#!/bin/bash

# ============================================================================
# EDIT THESE PARAMETERS GLOBALLY - they will be used in both #SBATCH directives
# and in the bash script
# ============================================================================
JOB_NAME="bbdm_f4_test"
MEM="40G"
GPUS="rtx_2080:1"
# =====================================================================ß=======
# INFERRED VALUES - automatically derived from the parameters above
# ============================================================================
SCRIPT_FILE="$(pwd)/taming-transformers/main.py"
OUTPUT_FILE="$(pwd)/outputs/${JOB_NAME}.out"
SBATCH_FILE="$(pwd)/outputs/${JOB_NAME}.sbatch"

# ============================================================================
# Generate the SBATCH file with variables substituted
# ============================================================================
mkdir -p "$(pwd)/outputs"
cat > "${SBATCH_FILE}" << 'EOF'
#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH
#SBATCH --partition main ### partition name where to run a job. Use 'main' unless qos is required. qos partitions 'rtx3090' 'rtx2080' 'gtx1080'
#SBATCH --time 0-10:30:00 ### limit the time of job running. Make sure it is not greater than the partition time limit (7 days)!! Format: D-H:MM:SS
#SBATCH --job-name JOB_NAME_PLACEHOLDER ### name of the job. replace my_job with your desired job name
#SBATCH --output OUTPUT_FILE_PLACEHOLDER ### output log for running job - %J is the job number variable
#SBATCH --mail-user=avinoamd@post.bgu.ac.il ### user's email for sending job status notifications
#SBATCH --mail-type=BEGIN,END,FAIL ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=MEM_PLACEHOLDER
#SBATCH --gpus=GPUS_PLACEHOLDER ### number of GPUs. Choosing type e.g.: #SBATCH --gpus=gtx_1080:1 , or rtx_2080, or rtx_3090 . Allocating more than 1 requires the IT team's permission
#SBATCH --tasks=1 # 1 process – use for processing of few programs concurrently in a job (with srun). Use just 1 otherwise

### Print some data to output file ###
echo "#########################################################"
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "#########################################################"

### Start your code below ####
module load cuda/12.4
cd /home/avinoamd/roni
/home/avinoamd/.conda/envs/chess-proj/bin/python -m src.bbdm.main -c src/bbdm/configs/f4_config.yaml --sample_to_eval

echo "#########################################################"
echo "Script ended successfully"
echo "#########################################################"
EOF

# Replace placeholders with actual values
sed -i "s|JOB_NAME_PLACEHOLDER|${JOB_NAME}|g" "${SBATCH_FILE}"
sed -i "s|SCRIPT_FILE_PLACEHOLDER|${SCRIPT_FILE}|g" "${SBATCH_FILE}"
sed -i "s|OUTPUT_FILE_PLACEHOLDER|${OUTPUT_FILE}|g" "${SBATCH_FILE}"
sed -i "s|MEM_PLACEHOLDER|${MEM}|g" "${SBATCH_FILE}"
sed -i "s|GPUS_PLACEHOLDER|${GPUS}|g" "${SBATCH_FILE}"
sed -i "s|HF_TOKEN_PLACEHOLDER|${HF_TOKEN}|g" "${SBATCH_FILE}"

sbatch "${SBATCH_FILE}"
