add_library_unity(gpgomea_evolution OBJECT EvolutionRun.cpp EvolutionState.cpp GenerationHandler.cpp NSGA2GenerationHandler.cpp SPEA2GenerationHandler.cpp PopulationInitializer.cpp LengthControlTruncationGenerationHandler.cpp AlphaDominance.cpp NSGA2DPGenerationHandler.cpp)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:gpgomea_evolution>
    PARENT_SCOPE)

