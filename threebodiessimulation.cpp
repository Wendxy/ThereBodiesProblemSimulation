// need a
// 1. add a planet to the simulation
// 2. add a force to the planet
// 3. update the position of the planet
// 4. update the velocity of the planet
// 5. update the acceleration of the planet
// 6. update the force of the planet
// 7. update the position of the planet
// 8. update the velocity of the planet
// 9. update the acceleration of the planet
// 10. update the force of the planet

#include "planet.hpp"
#include <iostream>
#include <fstream>

int main() {
    // Create planets with initial positions
    Planet p1("Body1", 1.0, 1e26, Vector3D(0, 0, 0));
    Planet p2("Body2", 1.0, 1e26, Vector3D(1e8, 0, 0));
    Planet p3("Body3", 1.0, 1e26, Vector3D(0, 1e8, 0));
    
    // Create bodies with initial velocities
    Body body1(p1, Velocity(Vector3D(0, 1e3, 0)));
    Body body2(p2, Velocity(Vector3D(0, -5e2, 0)));
    Body body3(p3, Velocity(Vector3D(-5e2, 0, 0)));
    
    // Create simulation with time step
    double dt = 100.0;  // seconds
    ThreeBodySimulation sim(body1, body2, body3, dt);
    
    // Open output file
    std::ofstream outfile("simulation_data.csv");
    outfile << "time,body1_x,body1_y,body1_z,body2_x,body2_y,body2_z,body3_x,body3_y,body3_z\n";
    
    // Run simulation
    int numSteps = 10000;
    for (int i = 0; i < numSteps; i++) {
        sim.step();
        
        // Write data every step
        outfile << i * dt << ","
                << body1.getPlanet().getPosition().getX() << ","
                << body1.getPlanet().getPosition().getY() << ","
                << body1.getPlanet().getPosition().getZ() << ","
                << body2.getPlanet().getPosition().getX() << ","
                << body2.getPlanet().getPosition().getY() << ","
                << body2.getPlanet().getPosition().getZ() << ","
                << body3.getPlanet().getPosition().getX() << ","
                << body3.getPlanet().getPosition().getY() << ","
                << body3.getPlanet().getPosition().getZ() << "\n";
        
        if (i % 100 == 0) {
            std::cout << "Progress: " << (100.0 * i / numSteps) << "%\r" << std::flush;
        }
    }
    
    outfile.close();
    std::cout << "\nSimulation complete! Data saved to simulation_data.csv" << std::endl;
    
    return 0;
}