#CPLEX system and lib format
SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic

# CPLEX directory
CPLEXDIR      = ${CPLEX_DIRECTORY}/cplex
CONCERTDIR    = ${CPLEX_DIRECTORY}/concert

# ---------------------------------------------------------------------
# Compiler selection 
# ---------------------------------------------------------------------

CCC = clang++ -O0

# ---------------------------------------------------------------------
# Compiler options 
# ---------------------------------------------------------------------

CCOPT = -m64 -Wall -fPIC -Ofast -fexceptions -DIL_STD -stdlib=libstdc++
#-O

#  ---------------------------------------------------------------------
# Link options and libraries
# ---------------------------------------------------------------------
#libraries
CPLEXLIBDIR   = $(CPLEXDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
CONCERTLIBDIR = $(CONCERTDIR)/lib/$(SYSTEM)/$(LIBFORMAT)

#includes
CONCERTINCDIR = $(CONCERTDIR)/include
CPLEXINCDIR   = $(CPLEXDIR)/include

# ---------------------------------------------------------------------
# Source and include file directory
# ---------------------------------------------------------------------

EXDIR= ./classes


EXINC         = $(EXDIR)/include
EXOBJ	       = $(EXDIR)/obj
EXSRC         = $(EXDIR)/src

# ---------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------
#include flags
CCFLAGS = $(CCOPT) -I$(CPLEXINCDIR) -I$(CONCERTINCDIR) -I$(EXINC) -v


#library flags
CCLNFLAGS = -L$(CPLEXLIBDIR) -lilocplex -lcplex -L$(CONCERTLIBDIR) -lconcert -lm -pthread

$(info $$CPLEX_DIRECTORY is [${CPLEX_DIRECTORY}])
$(info $$CPLEXDIR is [${CPLEXDIR}])
$(info $$CPLEXINCDIR is [${CPLEXINCDIR}])
$(info $$CPLEXLIBDIR is [${CPLEXLIBDIR}])

main: main.cpp DefiningPoint.o
	${CCC} ${CCFLAGS} main.cpp $(EXOBJ)/DefiningPoint.o ${CCLNFLAGS} -o main 

DefiningPoint.o:
	$(CCC) -c $(CCFLAGS) $(EXSRC)/DefiningPoint.cpp -o $(EXOBJ)/DefiningPoint.o




