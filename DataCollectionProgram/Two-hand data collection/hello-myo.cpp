// Copyright (C) 2013-2014 Thalmic Labs Inc.
// Distributed under the Myo SDK license agreement. See LICENSE.txt for details.
// This sample illustrates how to log EMG and IMU data. EMG streaming is only supported for one Myo at a time, and this entire sample is geared to one armband
#define _USE_MATH_DEFINES
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <time.h>
#include <myo/myo.hpp>
#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <vector>

HANDLE hSemaphore;
DWORD WINAPI Thread1Proc(LPVOID lpParamter);
DWORD WINAPI Thread2Proc(LPVOID lpParamter);

class DataCollector : public myo::DeviceListener {
private:
	std::string userName;
	std::string gesName;
	std::string times;
	std::string No="Myo1";
	std::vector<myo::Myo*> knownMyos;
	bool hasCreateFile = false;

public:
	// The files we are logging to
	std::ofstream emgFile;
	std::ofstream gyroFile;
	std::ofstream orientationFile;
	std::ofstream orientationEulerFile;
	std::ofstream accelerometerFile;
	std::ofstream emgFile2;
	std::ofstream gyroFile2;
	std::ofstream orientationFile2;
	std::ofstream orientationEulerFile2;
	std::ofstream accelerometerFile2;
	//============================================================================
	DataCollector(std::string uN, std::string gesN, std::string ti)
	{
		userName = uN;
		gesName = gesN;
		times = ti;
		openFiles();
	}
	//============================================================================
	size_t identifyMyo(myo::Myo* myo) {
		// Walk through the list of Myo devices that we've seen pairing events for.
		for (size_t i = 0; i < knownMyos.size(); ++i) {
			// If two Myo pointers compare equal, they refer to the same Myo device.
			if (knownMyos[i] == myo) {
				return i + 1;
			}
		}
		return 0;
	}
	void onPair(myo::Myo* myo, uint64_t timestamp, myo::FirmwareVersion firmwareVersion)
	{
		// Print out the MAC address of the armband we paired with.

		// The pointer address we get for a Myo is unique - in other words, it's safe to compare two Myo pointers to
		// see if they're referring to the same Myo.

		// Add the Myo pointer to our list of known Myo devices. This list is used to implement identifyMyo() below so
		// that we can give each Myo a nice short identifier.
		knownMyos.push_back(myo);

		// Now that we've added it to our list, get our short ID for it and print it out.
		std::cout << "Paired with " << identifyMyo(myo) << "." << std::endl;
	}
	void onConnect(myo::Myo *myo, uint64_t timestamp, myo::FirmwareVersion firmwareVersion) {
		//Reneable streaming
		std::cout << "Myo " << identifyMyo(myo) << " has connected." << std::endl;
		myo->setStreamEmg(myo::Myo::streamEmgEnabled);
	}
	//============================================================================
	void openFiles() {
		time_t timestamp = std::time(0);

		// Open file for EMG log
		if (emgFile.is_open()) {
			emgFile.close();
		}
		std::ostringstream emgFileString;
		emgFileString << No << "-" << userName << "-" << gesName << "-" << times << "-emg-" << timestamp << ".csv";
		emgFile.open(emgFileString.str(), std::ios::out);
		emgFile << "timestamp,emg1,emg2,emg3,emg4,emg5,emg6,emg7,emg8" << std::endl;

		// Open file for gyroscope log
		if (gyroFile.is_open()) {
			gyroFile.close();
		}
		std::ostringstream gyroFileString;
		gyroFileString << No << "-" << userName << "-" << gesName << "-" << times << "-gyro-" << timestamp << ".csv";
		gyroFile.open(gyroFileString.str(), std::ios::out);
		gyroFile << "timestamp,x,y,z" << std::endl;

		// Open file for accelerometer log
		if (accelerometerFile.is_open()) {
			accelerometerFile.close();
		}
		std::ostringstream accelerometerFileString;
		accelerometerFileString << No << "-" << userName << "-" << gesName << "-" << times << "-accelerometer-" << timestamp << ".csv";
		accelerometerFile.open(accelerometerFileString.str(), std::ios::out);
		accelerometerFile << "timestamp,x,y,z" << std::endl;

		// Open file for orientation log
		if (orientationFile.is_open()) {
			orientationFile.close();
		}
		std::ostringstream orientationFileString;
		orientationFileString << No << "-" << userName << "-" << gesName << "-" << times << "-orientation-" << timestamp << ".csv";
		orientationFile.open(orientationFileString.str(), std::ios::out);
		orientationFile << "timestamp,x,y,z,w" << std::endl;

		// Open file for orientation (Euler angles) log
		if (orientationEulerFile.is_open()) {
			orientationEulerFile.close();
		}
		std::ostringstream orientationEulerFileString;
		orientationEulerFileString << No << "-" << userName << "-" << gesName << "-" << times << "-orientationEuler-" << timestamp << ".csv";
		orientationEulerFile.open(orientationEulerFileString.str(), std::ios::out);
		orientationEulerFile << "timestamp,roll,pitch,yaw" << std::endl;
		//2=============================================================================
		// Open file for EMG log
		No = "Myo2";
		if (emgFile2.is_open()) {
			emgFile2.close();
		}
		std::ostringstream emgFileString2;
		emgFileString2 << No << "-" << userName << "-" << gesName << "-" << times << "-emg-" << timestamp << ".csv";
		emgFile2.open(emgFileString2.str(), std::ios::out);
		emgFile2 << "timestamp,emg1,emg2,emg3,emg4,emg5,emg6,emg7,emg8" << std::endl;

		// Open file for gyroscope log
		if (gyroFile2.is_open()) {
			gyroFile2.close();
		}
		std::ostringstream gyroFileString2;
		gyroFileString2 << No << "-" << userName << "-" << gesName << "-" << times << "-gyro-" << timestamp << ".csv";
		gyroFile2.open(gyroFileString2.str(), std::ios::out);
		gyroFile2 << "timestamp,x,y,z" << std::endl;

		// Open file for accelerometer log
		if (accelerometerFile2.is_open()) {
			accelerometerFile2.close();
		}
		std::ostringstream accelerometerFileString2;
		accelerometerFileString2 << No << "-" << userName << "-" << gesName << "-" << times << "-accelerometer-" << timestamp << ".csv";
		accelerometerFile2.open(accelerometerFileString2.str(), std::ios::out);
		accelerometerFile2 << "timestamp,x,y,z" << std::endl;

		// Open file for orientation log
		if (orientationFile2.is_open()) {
			orientationFile2.close();
		}
		std::ostringstream orientationFileString2;
		orientationFileString2 << No << "-" << userName << "-" << gesName << "-" << times << "-orientation-" << timestamp << ".csv";
		orientationFile2.open(orientationFileString2.str(), std::ios::out);
		orientationFile2 << "timestamp,x,y,z,w" << std::endl;

		// Open file for orientation (Euler angles) log
		if (orientationEulerFile2.is_open()) {
			orientationEulerFile2.close();
		}
		std::ostringstream orientationEulerFileString2;
		orientationEulerFileString2 << No << "-" << userName << "-" << gesName << "-" << times << "-orientationEuler-" << timestamp << ".csv";
		orientationEulerFile2.open(orientationEulerFileString2.str(), std::ios::out);
		orientationEulerFile2 << "timestamp,roll,pitch,yaw" << std::endl;

	}

	// onEmgData() is called whenever a paired Myo has provided new EMG data, and EMG streaming is enabled.
	void onEmgData(myo::Myo* myo, uint64_t timestamp, const int8_t* emg)
	{
		if (identifyMyo(myo)==1)
		{
			emgFile << timestamp;
			for (size_t i = 0; i < 8; i++) {
				emgFile << ',' << static_cast<int>(emg[i]);

			}
			emgFile << std::endl;
		}
		else
		{
			emgFile2 << timestamp;
			for (size_t i = 0; i < 8; i++) {
				emgFile2 << ',' << static_cast<int>(emg[i]);

			}
			emgFile2 << std::endl;
		}
		
	}

	// onOrientationData is called whenever new orientation data is provided
	// Be warned: This will not make any distiction between data from other Myo armbands
	void onOrientationData(myo::Myo *myo, uint64_t timestamp, const myo::Quaternion< float > &rotation) {
		if (identifyMyo(myo)==1)
		{
			orientationFile << timestamp
				<< ',' << rotation.x()
				<< ',' << rotation.y()
				<< ',' << rotation.z()
				<< ',' << rotation.w()
				<< std::endl;

			using std::atan2;
			using std::asin;
			using std::sqrt;
			using std::max;
			using std::min;

			// Calculate Euler angles (roll, pitch, and yaw) from the unit quaternion.
			float roll = atan2(2.0f * (rotation.w() * rotation.x() + rotation.y() * rotation.z()),
				1.0f - 2.0f * (rotation.x() * rotation.x() + rotation.y() * rotation.y()));
			float pitch = asin(max(-1.0f, min(1.0f, 2.0f * (rotation.w() * rotation.y() - rotation.z() * rotation.x()))));
			float yaw = atan2(2.0f * (rotation.w() * rotation.z() + rotation.x() * rotation.y()),
				1.0f - 2.0f * (rotation.y() * rotation.y() + rotation.z() * rotation.z()));

			orientationEulerFile << timestamp
				<< ',' << roll
				<< ',' << pitch
				<< ',' << yaw
				<< std::endl;
		}
		else
		{
			orientationFile2 << timestamp
				<< ',' << rotation.x()
				<< ',' << rotation.y()
				<< ',' << rotation.z()
				<< ',' << rotation.w()
				<< std::endl;

			using std::atan2;
			using std::asin;
			using std::sqrt;
			using std::max;
			using std::min;

			// Calculate Euler angles (roll, pitch, and yaw) from the unit quaternion.
			float roll = atan2(2.0f * (rotation.w() * rotation.x() + rotation.y() * rotation.z()),
				1.0f - 2.0f * (rotation.x() * rotation.x() + rotation.y() * rotation.y()));
			float pitch = asin(max(-1.0f, min(1.0f, 2.0f * (rotation.w() * rotation.y() - rotation.z() * rotation.x()))));
			float yaw = atan2(2.0f * (rotation.w() * rotation.z() + rotation.x() * rotation.y()),
				1.0f - 2.0f * (rotation.y() * rotation.y() + rotation.z() * rotation.z()));

			orientationEulerFile2 << timestamp
				<< ',' << roll
				<< ',' << pitch
				<< ',' << yaw
				<< std::endl;
		}
		
	}

	// onAccelerometerData is called whenever new acceleromenter data is provided
	// Be warned: This will not make any distiction between data from other Myo armbands
	void onAccelerometerData(myo::Myo *myo, uint64_t timestamp, const myo::Vector3< float > &accel) {
		if (identifyMyo(myo)==1)
		{
			printVector(accelerometerFile, timestamp, accel);
		}
		else {
			printVector(accelerometerFile2, timestamp, accel);
		}
	}

	// onGyroscopeData is called whenever new gyroscope data is provided
	// Be warned: This will not make any distiction between data from other Myo armbands
	void onGyroscopeData(myo::Myo *myo, uint64_t timestamp, const myo::Vector3< float > &gyro) {
		if (identifyMyo(myo) == 1)
		{
			printVector(gyroFile, timestamp, gyro);
		}
		else {
			printVector(gyroFile2, timestamp, gyro);
		}
	}

	// Helper to print out accelerometer and gyroscope vectors
	void printVector(std::ofstream &file, uint64_t timestamp, const myo::Vector3< float > &vector) {
		file << timestamp
			<< ',' << vector.x()
			<< ',' << vector.y()
			<< ',' << vector.z()
			<< std::endl;
	}
	

};

int main(int argc, char** argv)
{
	HANDLE hThread1;
	HANDLE hThread2;
	hSemaphore = CreateSemaphore(NULL, 1, 1, NULL);      //创建信号量，初始为1，最多为1  
	hThread2 = CreateThread(NULL, 0, Thread2Proc, NULL, 0, NULL);   //创建线程
	hThread1 = CreateThread(NULL, 0, Thread1Proc, NULL, 0, NULL);   //创建线程
	HANDLE aThread[2];
	aThread[0] = hThread1;
	aThread[1] = hThread2;
	WaitForMultipleObjects(2, aThread, TRUE, INFINITE);
	CloseHandle(hThread2);
	CloseHandle(hThread1);       //释放句柄
	CloseHandle(hSemaphore);
}

DWORD WINAPI Thread1Proc(LPVOID lpParamter)
{
	char ch;
	bool key = 1;
	while (1) {
		if (_kbhit()) {
			ch = _getch();
			if (ch == 13) {
				if (key) {
					WaitForSingleObject(hSemaphore, INFINITE);
					key = !key;
					std::cout << "PAUSE\n";
				}
				else
				{
					ReleaseSemaphore(hSemaphore, 1, NULL);
					key = !key;
					std::cout << "CONTINUE\n";
				}
			}
			else if (ch = 27) {
				exit(0);
			}
		}
	}
	return 0L;
}

DWORD WINAPI Thread2Proc(LPVOID lpParamter)
{
	std::string uN, gesN, ti;
	// We catch any exceptions that might occur below -- see the catch statement for more details.
	try {

		// First, we create a Hub with our application identifier. Be sure not to use the com.example namespace when
		// publishing your application. The Hub provides access to one or more Myos.
		myo::Hub hub("com.undercoveryeti.myo-data-capture");

		std::cout << "Attempting to find a Myo..." << std::endl;
		/*
		// Next, we attempt to find a Myo to use. If a Myo is already paired in Myo Connect, this will return that Myo
		// immediately.
		// waitForMyo() takes a timeout value in milliseconds. In this case we will try to find a Myo for 10 seconds, and
		// if that fails, the function will return a null pointer.
		myo::Myo* myo = hub.waitForMyo(10000);

		// If waitForMyo() returned a null pointer, we failed to find a Myo, so exit with an error message.
		if (!myo) {
			throw std::runtime_error("Unable to find Myo!");
		}

		// We've found a Myo.
		std::cout << "Connected to  Myo armband! Logging to the file system. \n";
		// Next we enable EMG streaming on the found Myo.
		myo->setStreamEmg(myo::Myo::streamEmgEnabled);
		*/
		std::cout << "Pleaze insert your name,gesture name,times for the file.\n" << std::endl;
		std::cin >> uN >> gesN >> ti;

		// Next we construct an instance of our DeviceListener, so that we can register it with the Hub.
		DataCollector collector(uN, gesN, ti);
		

		// Hub::addListener() takes the address of any object whose class inherits from DeviceListener, and will cause
		// Hub::run() to send events to all registered device listeners.
		hub.addListener(&collector);
		std::cout << "Now start...\n";
		int k = 1;
		// Finally we enter our main loop.
		while (true) {
			// In each iteration of our main loop, we run the Myo event loop for a set number of milliseconds.
			// In this case, we wish to update our display 50 times a second, so we run for 1000/20 milliseconds.
			WaitForSingleObject(hSemaphore, INFINITE);
			hub.run(1);
			k++;
			if ((k % 1500) == 0)
			{
				std::cout << ".";
			}
			ReleaseSemaphore(hSemaphore, 1, NULL);
		}
		// If a standard exception occurred, we print out its message and exit.
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		std::cerr << "Press enter to continue.";
		std::cin.ignore();
		return 1L;
	}
	return 0L;
}
