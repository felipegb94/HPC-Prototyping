# Charm++
This repository contains different code sample that make use of Charm++. 
If running in euler or a development node make sure you run the following two commands before trying to build anything with charm++:

```
    module load gcc
    module load charm
```

## Charm++ Local setup in Linux/OSX
To setup Charm++ in a Unix based system it was relatively simple. Either clone the git repository or download the distribution version and then go into the folder and run:`./build`. Answer the different question the build program asks you. Then add the following line to you bash file, in OSX I added the line to `.bash_profile`:

```
export CHARMDIR = "PATH TO THE DIRECTORY WHERE YOU RAN ./build"
```

This `$CHARMDIR` environment variable will be used by the Makefiles in your Charm++ program.

## Common 
To compile any of the programs in this folder you need to make sure you have gcc and add the following line in the Makefile:

```
CHARMDIR = [Insert the path to your charm installation directiory]
```

If you followied the local setup described above, then there is no need to declare the CHARMDIR variable since it already is an environment variable. Therefore the following line should have the right values

```
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)
```

## ArrayHello
### Elements of the Program
#### Main Chare
Purpose of the Main chare is to: Create the other chare objects in the application, initiate the computation, call CkExit() when the calculation is finished.

#### Hello Chare

## hello

Prints hello world using the charm++ file structure. 

After editing the Makefile you should be able to build and run the hello world with the following commands:

```
make
./charmrun ./hello
```

The binary charmrun the Charm++ runtime environment.



