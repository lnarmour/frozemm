// taken and modified from here:
// https://github.com/kajalv/nvml-power

#include <cstdlib>
#include <chrono>
#include "nvmlPower.hpp"

/*
These may be encompassed in a class if desired. Trivial CUDA programs written for the purpose of benchmarking might prefer this approach.
*/
bool pollThreadStatus = false;
unsigned int deviceCount = 0;
char deviceNameStr[64];
float total_energy;
long total_time;

nvmlReturn_t nvmlResult;
nvmlDevice_t nvmlDeviceID;
nvmlPciInfo_t nvmPCIInfo;
nvmlEnableState_t pmmode;
nvmlComputeMode_t computeMode;

pthread_t powerPollThread;

/*
Poll the GPU using nvml APIs.
*/
void *powerPollingFunc(void *ptr)
{
	unsigned int powerLevel = 0;

  std::chrono::milliseconds timestamp;
  timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()
  );
  long start_time = timestamp.count();
  long elapsed_time = 0;
  long curr_time;
  long delta_time;
  float energy = 0.0;

	while (pollThreadStatus)
	{
    timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now().time_since_epoch()
    );
    curr_time = timestamp.count();
    delta_time = curr_time - (elapsed_time + start_time);
    if (delta_time > 4) {
      elapsed_time = curr_time - start_time;
    } else {
      continue;
    }   

		pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);
		// Get the power management mode of the GPU.
		nvmlResult = nvmlDeviceGetPowerManagementMode(nvmlDeviceID, &pmmode);

		// The following function may be utilized to handle errors as needed.
		getNVMLError(nvmlResult);

		// Check if power management mode is enabled.
		if (pmmode == NVML_FEATURE_ENABLED)
		{
		 	// Get the power usage in milliWatts.
		 	nvmlResult = nvmlDeviceGetPowerUsage(nvmlDeviceID, &powerLevel);
		}

    if (std::getenv("VERBOSE")) {
      printf("%d,%d\n", elapsed_time, powerLevel);
    }
    energy += (delta_time / 1000.0) * (powerLevel / 1000.0);
		pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
	}
  total_time = elapsed_time;
  total_energy = energy;
  //printf("energy:     %.2f Joules\n", energy);
	pthread_exit(0);
}

/*
Start power measurement by spawning a pthread that polls the GPU.
Function needs to be modified as per usage to handle errors as seen fit.
*/
void nvmlAPIRun()
{
	int i;

	// Initialize nvml.
	nvmlResult = nvmlInit();
	if (NVML_SUCCESS != nvmlResult)
	{
		printf("NVML Init fail: %s\n", nvmlErrorString(nvmlResult));
		exit(0);
	}

	// Count the number of GPUs available.
	nvmlResult = nvmlDeviceGetCount(&deviceCount);
	if (NVML_SUCCESS != nvmlResult)
	{
		printf("Failed to query device count: %s\n", nvmlErrorString(nvmlResult));
		exit(0);
	}

	for (i = 0; i < deviceCount; i++)
	{
		// Get the device ID.
		nvmlResult = nvmlDeviceGetHandleByIndex(i, &nvmlDeviceID);
		if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get handle for device %d: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}

		// Get the name of the device.
		nvmlResult = nvmlDeviceGetName(nvmlDeviceID, deviceNameStr, sizeof(deviceNameStr)/sizeof(deviceNameStr[0]));
		if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get name of device %d: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}

		// Get PCI information of the device.
		nvmlResult = nvmlDeviceGetPciInfo(nvmlDeviceID, &nvmPCIInfo);
		if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get PCI info of device %d: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}

		// Get the compute mode of the device which indicates CUDA capabilities.
		nvmlResult = nvmlDeviceGetComputeMode(nvmlDeviceID, &computeMode);
		if (NVML_ERROR_NOT_SUPPORTED == nvmlResult)
		{
			printf("This is not a CUDA-capable device.\n");
		}
		else if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get compute mode for device %i: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}
	}

	// This statement assumes that the first indexed GPU will be used.
	// If there are multiple GPUs that can be used by the system, this needs to be done with care.
	// Test thoroughly and ensure the correct device ID is being used.
	nvmlResult = nvmlDeviceGetHandleByIndex(0, &nvmlDeviceID);

	pollThreadStatus = true;

	const char *message = "Test";
	int iret = pthread_create(&powerPollThread, NULL, powerPollingFunc, (void*) message);
	if (iret)
	{
		fprintf(stderr,"Error - pthread_create() return code: %d\n",iret);
		exit(0);
	}
}

/*
End power measurement. This ends the polling thread.
*/
void nvmlAPIEnd()
{
	pollThreadStatus = false;
	pthread_join(powerPollThread, NULL);

	nvmlResult = nvmlShutdown();
	if (NVML_SUCCESS != nvmlResult)
	{
		printf("Failed to shut down NVML: %s\n", nvmlErrorString(nvmlResult));
		exit(0);
	}
}

long nvmlAPI_getTotalTime() {
  return total_time;
}

float nvmlAPI_getEnergy() {
  return total_energy;
}

/*
Return a number with a specific meaning. This number needs to be interpreted and handled appropriately.
*/
int getNVMLError(nvmlReturn_t resultToCheck)
{
	if (resultToCheck == NVML_ERROR_UNINITIALIZED)
		return 1;
	if (resultToCheck == NVML_ERROR_INVALID_ARGUMENT)
		return 2;
	if (resultToCheck == NVML_ERROR_NOT_SUPPORTED)
		return 3;
	if (resultToCheck == NVML_ERROR_NO_PERMISSION)
		return 4;
	if (resultToCheck == NVML_ERROR_ALREADY_INITIALIZED)
		return 5;
	if (resultToCheck == NVML_ERROR_NOT_FOUND)
		return 6;
	if (resultToCheck == NVML_ERROR_INSUFFICIENT_SIZE)
		return 7;
	if (resultToCheck == NVML_ERROR_INSUFFICIENT_POWER)
		return 8;
	if (resultToCheck == NVML_ERROR_DRIVER_NOT_LOADED)
		return 9;
	if (resultToCheck == NVML_ERROR_TIMEOUT)
		return 10;
	if (resultToCheck == NVML_ERROR_IRQ_ISSUE)
		return 11;
	if (resultToCheck == NVML_ERROR_LIBRARY_NOT_FOUND)
		return 12;
	if (resultToCheck == NVML_ERROR_FUNCTION_NOT_FOUND)
		return 13;
	if (resultToCheck == NVML_ERROR_CORRUPTED_INFOROM)
		return 14;
	if (resultToCheck == NVML_ERROR_GPU_IS_LOST)
		return 15;
	if (resultToCheck == NVML_ERROR_UNKNOWN)
		return 16;

	return 0;
}
