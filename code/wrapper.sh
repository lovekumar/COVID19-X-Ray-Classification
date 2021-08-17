action=$1
JOB_NAME=$2

env_setup()
{
  env_details=`conda env list | grep 'team11-pyspark'`
  if [ -z "${env_details}" ]
  then
    echo "Setting Up Environment first time"
	  conda env create -f environment.yml
	else
	  echo "Environment already exist, updating environment"
    conda env update -f environment.yml
  fi
}

run_job()
{

	echo "START Time: `date`"
	export TF_CPP_MIN_LOG_LEVEL="2"
	export KMP_DUPLICATE_LIB_OK="True"

	zip_file=src.zip
	sparkld_folder=databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11
	sparkld_tar=resources/sparkld_1.5.0-spark2.4-s_2.11.tgz


	# cleanup
	find . -name '__pycache__' | xargs rm -rf
	rm -f ${zip_file}
	rm -rf ${sparkld_folder}

	# create the zip
	zip -r ${zip_file} src/

	# Untar sparkld
	tar -zxf ${sparkld_tar}

  # run the job
	spark-submit --driver-memory 12g \
	             --packages ${sparkld_folder} \
	             --py-files ${zip_file} src/main.py \
	             --job $JOB_NAME

	echo "END Time: `date`"
}


case $action in
  "run")
     run_job
     ;;
  "env-setup")
     env_setup
     ;;
  *)
     echo "Not Valid Input"
     ;;
esac
