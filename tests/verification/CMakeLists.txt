# V&V cases from the project specification.
softflow_add_test(test_poiseuille      test_poiseuille.cpp)
softflow_add_test(test_poiseuille_collision_operators test_poiseuille_collision_operators.cpp)
softflow_add_test(test_membrane_constitutive test_membrane_constitutive.cpp)
softflow_add_test(test_taylor_green    test_taylor_green.cpp)
softflow_add_test(test_pure_diffusion    test_pure_diffusion.cpp)
softflow_add_test(test_ibm_momentum_conservation test_ibm_momentum_conservation.cpp)
softflow_add_test(test_chemistry_mass_balance    test_chemistry_mass_balance.cpp)
softflow_add_test(test_lubrication_brenner       test_lubrication_brenner.cpp)
softflow_add_test(test_bell_adhesion_kinetics    test_bell_adhesion_kinetics.cpp)
softflow_add_test(test_capsule_shear             test_capsule_shear.cpp)
softflow_add_test(test_segre_silberberg          test_segre_silberberg.cpp)
softflow_add_test(test_capsule_shear_convergence test_capsule_shear_convergence.cpp)
# Re-enabled for: rewritten with circular-statistics
# (Mardia & Jupp) treatment of the centroid + variance on the periodic ring,
# which removes the periodic-wrap pollution that previously parked the test.
softflow_add_test(test_taylor_dispersion test_taylor_dispersion.cpp)
