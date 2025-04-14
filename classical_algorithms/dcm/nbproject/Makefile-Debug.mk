#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=GNU-Linux-x86
CND_DLIB_EXT=so
CND_CONF=Debug
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/Sources/DCM_Solver.o \
	${OBJECTDIR}/Sources/DCM_nodes.o \
	${OBJECTDIR}/Sources/Infeasible_Elements.o \
	${OBJECTDIR}/Sources/MDCM_Solver.o \
	${OBJECTDIR}/Sources/MDCM_nodes.o \
	${OBJECTDIR}/Sources/NDPs.o \
	${OBJECTDIR}/Sources/Nadir_Solver.o \
	${OBJECTDIR}/Sources/Resetting_functions.o \
	${OBJECTDIR}/Sources/Y_tilde_class.o \
	${OBJECTDIR}/main.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=/software/opt-sw/cplex-1210/concert/lib/x86-64_linux/static_pic/libconcert.a /software/opt-sw/cplex-1210/cplex/lib/x86-64_linux/static_pic/libilocplex.a /software/opt-sw/cplex-1210/cplex/lib/x86-64_linux/static_pic/libcplex.a -lpthread

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk DCM

DCM: /software/opt-sw/cplex-1210/concert/lib/x86-64_linux/static_pic/libconcert.a

DCM: /software/opt-sw/cplex-1210/cplex/lib/x86-64_linux/static_pic/libilocplex.a

DCM: /software/opt-sw/cplex-1210/cplex/lib/x86-64_linux/static_pic/libcplex.a

DCM: ${OBJECTFILES}
	${LINK.cc} -o DCM ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/Sources/DCM_Solver.o: Sources/DCM_Solver.cpp 
	${MKDIR} -p ${OBJECTDIR}/Sources
	${RM} $@.d
	$(COMPILE.cc) -g -DIL_STD -I/software/opt-sw/cplex-1210/cplex/include -I/software/opt-sw/cplex-1210/concert/include -IHeaders -MMD -MP -MF $@.d -o ${OBJECTDIR}/Sources/DCM_Solver.o Sources/DCM_Solver.cpp

${OBJECTDIR}/Sources/DCM_nodes.o: Sources/DCM_nodes.cpp 
	${MKDIR} -p ${OBJECTDIR}/Sources
	${RM} $@.d
	$(COMPILE.cc) -g -DIL_STD -I/software/opt-sw/cplex-1210/cplex/include -I/software/opt-sw/cplex-1210/concert/include -IHeaders -MMD -MP -MF $@.d -o ${OBJECTDIR}/Sources/DCM_nodes.o Sources/DCM_nodes.cpp

${OBJECTDIR}/Sources/Infeasible_Elements.o: Sources/Infeasible_Elements.cpp 
	${MKDIR} -p ${OBJECTDIR}/Sources
	${RM} $@.d
	$(COMPILE.cc) -g -DIL_STD -I/software/opt-sw/cplex-1210/cplex/include -I/software/opt-sw/cplex-1210/concert/include -IHeaders -MMD -MP -MF $@.d -o ${OBJECTDIR}/Sources/Infeasible_Elements.o Sources/Infeasible_Elements.cpp

${OBJECTDIR}/Sources/MDCM_Solver.o: Sources/MDCM_Solver.cpp 
	${MKDIR} -p ${OBJECTDIR}/Sources
	${RM} $@.d
	$(COMPILE.cc) -g -DIL_STD -I/software/opt-sw/cplex-1210/cplex/include -I/software/opt-sw/cplex-1210/concert/include -IHeaders -MMD -MP -MF $@.d -o ${OBJECTDIR}/Sources/MDCM_Solver.o Sources/MDCM_Solver.cpp

${OBJECTDIR}/Sources/MDCM_nodes.o: Sources/MDCM_nodes.cpp 
	${MKDIR} -p ${OBJECTDIR}/Sources
	${RM} $@.d
	$(COMPILE.cc) -g -DIL_STD -I/software/opt-sw/cplex-1210/cplex/include -I/software/opt-sw/cplex-1210/concert/include -IHeaders -MMD -MP -MF $@.d -o ${OBJECTDIR}/Sources/MDCM_nodes.o Sources/MDCM_nodes.cpp

${OBJECTDIR}/Sources/NDPs.o: Sources/NDPs.cpp 
	${MKDIR} -p ${OBJECTDIR}/Sources
	${RM} $@.d
	$(COMPILE.cc) -g -DIL_STD -I/software/opt-sw/cplex-1210/cplex/include -I/software/opt-sw/cplex-1210/concert/include -IHeaders -MMD -MP -MF $@.d -o ${OBJECTDIR}/Sources/NDPs.o Sources/NDPs.cpp

${OBJECTDIR}/Sources/Nadir_Solver.o: Sources/Nadir_Solver.cpp 
	${MKDIR} -p ${OBJECTDIR}/Sources
	${RM} $@.d
	$(COMPILE.cc) -g -DIL_STD -I/software/opt-sw/cplex-1210/cplex/include -I/software/opt-sw/cplex-1210/concert/include -IHeaders -MMD -MP -MF $@.d -o ${OBJECTDIR}/Sources/Nadir_Solver.o Sources/Nadir_Solver.cpp

${OBJECTDIR}/Sources/Resetting_functions.o: Sources/Resetting_functions.cpp 
	${MKDIR} -p ${OBJECTDIR}/Sources
	${RM} $@.d
	$(COMPILE.cc) -g -DIL_STD -I/software/opt-sw/cplex-1210/cplex/include -I/software/opt-sw/cplex-1210/concert/include -IHeaders -MMD -MP -MF $@.d -o ${OBJECTDIR}/Sources/Resetting_functions.o Sources/Resetting_functions.cpp

${OBJECTDIR}/Sources/Y_tilde_class.o: Sources/Y_tilde_class.cpp 
	${MKDIR} -p ${OBJECTDIR}/Sources
	${RM} $@.d
	$(COMPILE.cc) -g -DIL_STD -I/software/opt-sw/cplex-1210/cplex/include -I/software/opt-sw/cplex-1210/concert/include -IHeaders -MMD -MP -MF $@.d -o ${OBJECTDIR}/Sources/Y_tilde_class.o Sources/Y_tilde_class.cpp

${OBJECTDIR}/main.o: main.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -DIL_STD -I/software/opt-sw/cplex-1210/cplex/include -I/software/opt-sw/cplex-1210/concert/include -IHeaders -MMD -MP -MF $@.d -o ${OBJECTDIR}/main.o main.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} DCM

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
