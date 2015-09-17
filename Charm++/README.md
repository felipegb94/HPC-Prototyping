# Charm++
This repository contains different code sample that make use of Charm++. 
If running in euler or a development node make sure you run the following two commands before trying to build anything with charm++:

```
    module load gcc
    module load charm
```

## hello

Prints hello world using the charm++ file structure. To compile you need to make sure you have gcc and add the following line in the Makefile:

```
    CHARMDIR = [Insert the path to your charm installation directiory]
```

