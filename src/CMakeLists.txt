set(SOFTFLOW_SOURCES
    core/parameters.cpp
    core/insertion/region.cpp
    core/insertion/size_distribution.cpp
    core/insertion/inserter.cpp
    core/insertion/lattice_inserter.cpp
    core/insertion/rsa_inserter.cpp
    core/insertion/dynamic_inserter.cpp
    core/insertion/image_mask_region.cpp
    membrane/capsule.cpp
    membrane/capsule_system.cpp
    membrane/repulsion.cpp
    membrane/lubrication.cpp
    membrane/adhesion.cpp
    lbm/lattice_field.cpp
    lbm/bgk_collision.cpp
    lbm/mrt_collision.cpp
    lbm/regularized_bgk.cpp
    lbm/zou_he_boundary.cpp
    lbm/bounce_back.cpp
    lbm/moving_wall.cpp
    lbm/interpolated_bounce_back.cpp
    lbm/advection_diffusion.cpp
    lbm/free_surface.cpp
    lbm/lbm_solver.cpp
    lbm/shan_chen.cpp
    coupling/immersed_boundary.cpp
    coupling/viscosity_field.cpp
    geometry/channel_builder.cpp
    geometry/polygon_obstacle.cpp
    analysis/segregation_metrics.cpp
    ml/simple_nn.cpp
    ml/training_data.cpp
    io/vtk_writer.cpp
    io/csv_writer.cpp
    io/fluid_probe_writer.cpp
    io/global_stats_writer.cpp
    io/checkpoint.cpp
    engine/simulation.cpp
)

add_library(softflow STATIC ${SOFTFLOW_SOURCES})
target_include_directories(softflow
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        # Generated build_info.h lives under build/generated/softflow/.
        # The engine includes it as <softflow/build_info.h>.
        ${CMAKE_BINARY_DIR}/generated)

if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(softflow PUBLIC OpenMP::OpenMP_CXX)
endif()
