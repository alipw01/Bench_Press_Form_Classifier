# Qualitative Assessment of the Bench Press Exercise using Wrist-Worn Inertial Sensors

A project for enabling both real-time and retrospective form classification of the bench press exercise, providing feedback with information regarding form and repetitions.

---

**AUTHOR:** Alastair Palmer-Walsh

---

## HARDWARE REQUIREMENTS

- To run this project, the following hardware must be owned:
  - MIMU22BL and MIMU4844 sensors, made by Inertial Elements
  - 2x Micro-USB to _ cables (where _ is the applicable connector for your device), used for connecting the sensors to your device
  - 2 available ports for connecting the sensors to your device

---

## INSTRUCTIONS FOR USE

- Within this project, there are two files which can be used to perform form classification and repetition counting, giving you feedback on the execution of the exercise:

### real_time_classification.py

- This file should be used for giving real-time feedback on the execution of the bench press exercise, providing ongoing feedback regarding your form and repetitions and suggesting any corrections which should be made. Once finished, various statistics regarding your performance will be outputted, indicating how well you performed and what improvements should be made next time.

- To use this file, the MIMU22BL and MIMU4844 sensors should be connected, and their device port numbers of the ports they are connected to should be inserted at the bottom of the code, as marked out. Once done, the file can be run. Upon execution of the file, feedback will be given as to the state of the sensors. If messages from both sensors have been displayed stating that they are ready, the *space bar* can be pressed to begin recording data from the sensors, and you can start executing the exercise. If any errors have been displayed, try reconnecting the sensors and trying again. Once started, ongoing feedback will be displayed; to terminate the code, the 'q' key should be pressed twice.

### retrospective_classification.py

- This file should be used for giving retrospective feedback on the execution of the bench press exercise, providing feedback regarding your form and repetitions during a previous workout - suggesting how it can be improved in the future. Once finished, various statistics regarding your performance will be outputted, indicating how well you performed and what improvements should be made next time.

- To use this file, the associated data files for each sensor (containing the data gathered prior to execution) should be inserted at the bottom of the code, as marked out. Once done, the file can be run. Upon execution of the file, the statistics regarding the inputted workout will be displayed, showing how you performed and what improvements can be made for next time.

---

## SENSOR SETUP

- When collecting data, the MIMU22BL and MIMU4844 sensors should be positioned on the left and right arms, respectively. Both sensors should be positioned on the outside of the forearm and situated adjacent to the wrist, with the connecting wires going towards the body. These sensors should be attached using a sensor holder or any other sufficient technique which can hold them tightly in position.

---

## DISPLAYED FEEDBACK

- All feedback produced by the files are given in the form of a written message displayed in the terminal.
