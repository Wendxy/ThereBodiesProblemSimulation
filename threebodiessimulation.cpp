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
#include <sstream>
#include <vector>

static bool parseVec3(const std::string& text, Vector3D& out) {
    std::stringstream ss(text);
    std::string item;
    std::vector<double> parts;
    while (std::getline(ss, item, ',')) {
        try {
            parts.push_back(std::stod(item));
        } catch (...) {
            return false;
        }
    }
    if (parts.size() != 3) {
        return false;
    }
    out = Vector3D(parts[0], parts[1], parts[2]);
    return true;
}

static void printUsage() {
    std::cout
        << "Usage: threebodiessimulation [options]\n"
        << "Options:\n"
        << "  --dt <seconds>            Time step (default: 100.0)\n"
        << "  --steps <count>           Number of steps (default: 10000)\n"
        << "  --out <path>              Output CSV path (default: simulation_data.csv)\n"
        << "  --m1 <mass> --m2 <mass> --m3 <mass>  Masses in kg\n"
        << "  --p1 x,y,z --p2 x,y,z --p3 x,y,z     Positions in meters\n"
        << "  --v1 x,y,z --v2 x,y,z --v3 x,y,z     Velocities in m/s\n";
}

int main(int argc, char** argv) {
    // Defaults: Sun/Earth/Moon system
    double m1 = 1.989e30;
    double m2 = 5.972e24;
    double m3 = 7.342e22;
    Vector3D p1_pos(0, 0, 0);
    Vector3D p2_pos(1.496e11, 0, 0);
    Vector3D p3_pos(1.496e11 + 3.844e8, 0, 0);
    Vector3D v1_vel(0, 0, 0);
    Vector3D v2_vel(0, 29780, 0);
    Vector3D v3_vel(0, 29780 + 1022, 0);
    double dt = 100.0;  // seconds
    int numSteps = 10000;
    std::string outPath = "simulation_data.csv";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto requireValue = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                printUsage();
                std::exit(1);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--dt") {
            dt = std::stod(requireValue(arg));
        } else if (arg == "--steps") {
            numSteps = std::stoi(requireValue(arg));
        } else if (arg == "--out") {
            outPath = requireValue(arg);
        } else if (arg == "--m1") {
            m1 = std::stod(requireValue(arg));
        } else if (arg == "--m2") {
            m2 = std::stod(requireValue(arg));
        } else if (arg == "--m3") {
            m3 = std::stod(requireValue(arg));
        } else if (arg == "--p1") {
            if (!parseVec3(requireValue(arg), p1_pos)) {
                std::cerr << "Invalid --p1 value\n";
                return 1;
            }
        } else if (arg == "--p2") {
            if (!parseVec3(requireValue(arg), p2_pos)) {
                std::cerr << "Invalid --p2 value\n";
                return 1;
            }
        } else if (arg == "--p3") {
            if (!parseVec3(requireValue(arg), p3_pos)) {
                std::cerr << "Invalid --p3 value\n";
                return 1;
            }
        } else if (arg == "--v1") {
            if (!parseVec3(requireValue(arg), v1_vel)) {
                std::cerr << "Invalid --v1 value\n";
                return 1;
            }
        } else if (arg == "--v2") {
            if (!parseVec3(requireValue(arg), v2_vel)) {
                std::cerr << "Invalid --v2 value\n";
                return 1;
            }
        } else if (arg == "--v3") {
            if (!parseVec3(requireValue(arg), v3_vel)) {
                std::cerr << "Invalid --v3 value\n";
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage();
            return 1;
        }
    }

    // Create planets with initial positions (radius, mass, position)
    Planet p1("Body1", 6.96e8, m1, p1_pos);
    Planet p2("Body2", 6.371e6, m2, p2_pos);
    Planet p3("Body3", 1.737e6, m3, p3_pos);

    // Create bodies with initial velocities
    Body body1(p1, Velocity(v1_vel));
    Body body2(p2, Velocity(v2_vel));
    Body body3(p3, Velocity(v3_vel));

    // Create simulation with time step
    ThreeBodySimulation sim(body1, body2, body3, dt);
    
    // Open output file
    std::ofstream outfile(outPath);
    outfile << "time,"
            << "body1_x,body1_y,body1_z,body1_vx,body1_vy,body1_vz,"
            << "body2_x,body2_y,body2_z,body2_vx,body2_vy,body2_vz,"
            << "body3_x,body3_y,body3_z,body3_vx,body3_vy,body3_vz\n";
    
    // Run simulation
    for (int i = 0; i < numSteps; i++) {
        sim.step();
        
        // Write data every step
        outfile << i * dt << ","
                << body1.getPlanet().getPosition().getX() << ","
                << body1.getPlanet().getPosition().getY() << ","
                << body1.getPlanet().getPosition().getZ() << ","
                << body1.getVelocity().getVelocity().getX() << ","
                << body1.getVelocity().getVelocity().getY() << ","
                << body1.getVelocity().getVelocity().getZ() << ","
                << body2.getPlanet().getPosition().getX() << ","
                << body2.getPlanet().getPosition().getY() << ","
                << body2.getPlanet().getPosition().getZ() << ","
                << body2.getVelocity().getVelocity().getX() << ","
                << body2.getVelocity().getVelocity().getY() << ","
                << body2.getVelocity().getVelocity().getZ() << ","
                << body3.getPlanet().getPosition().getX() << ","
                << body3.getPlanet().getPosition().getY() << ","
                << body3.getPlanet().getPosition().getZ() << ","
                << body3.getVelocity().getVelocity().getX() << ","
                << body3.getVelocity().getVelocity().getY() << ","
                << body3.getVelocity().getVelocity().getZ() << "\n";
        
        if (i % 100 == 0) {
            std::cout << "Progress: " << (100.0 * i / numSteps) << "%\r" << std::flush;
        }
    }
    
    outfile.close();
    std::cout << "\nSimulation complete! Data saved to simulation_data.csv" << std::endl;
    
    return 0;
}
