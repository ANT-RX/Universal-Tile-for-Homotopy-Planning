/**
 * Algoritmo HPPM (Homotopy Path Planning Method) - Motor Numérico Optimizado
 * ============================================================================
 * (La lógica geométrica y clusterización ahora la provee el orquestador Python)
 * Mejoras Activas:
 * 1. Regularización Levenberg-Marquardt en Matriz Jacobiana.
 * 2. Valor Absoluto Suavizado calibrado a 0.0001.
 * 3. Fijación forzada de pendientes óptimas (m1=1, m2=4).
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define VERDADERO 1
#define FALSO 0

// Límites físicos (Zero-Allocation)
#define MAX_OBSTACULOS 1000
#define MAX_PASOS_HISTORIAL 5000

// Constantes matemáticas y de seguridad
#define PI 3.14159265358979323846
#define EPSILON_SEGURIDAD 0.0001 
#define EPSILON_CUADRADO (EPSILON_SEGURIDAD * EPSILON_SEGURIDAD) 
#define LIMITE_CERO_MATRIZ 1e-15
#define LIMITE_CERO_NORMA 1e-12
#define FACTOR_AMORTIGUACION 1e-7 

// Constantes de sintonización del algoritmo
#define MAX_ITERACIONES_NEWTON 150
#define TOLERANCIA_ERROR_F 1e-7
#define TOLERANCIA_ERROR_F_CUADRADO (TOLERANCIA_ERROR_F * TOLERANCIA_ERROR_F)
#define UMBRAL_RETROCESO_ANGULAR 0.985
#define UMBRAL_SALIDA_RETROCESO 0.9

#define VENTANA_ESTANCAMIENTO 15       
#define MIN_DISTANCIA_PROGRESO 0.005   

#define LIMITAR_ACOS(x) ((x) > 1.0 ? 1.0 : ((x) < -1.0 ? -1.0 : (x)))

typedef struct {
    int num_obstaculos;         
    double radio_esfera_paso;         
    double pendiente_inicial; 
    double pendiente_final;           
    int max_pasos_permitidos;           
    double inicio_x, inicio_y;           
    double meta_x, meta_y;           
    
    double matriz_obstaculos[MAX_OBSTACULOS][4]; // [x, y, radio^2, peso_neto_procesado]           
    double historial_ruta[MAX_PASOS_HISTORIAL][3]; 
    int pasos_totales_dados;   
} EntornoNavegacion;

typedef struct {
    double x, y, L;                         
    double x_anterior, y_anterior, L_anterior; 
    double centro_x, centro_y, centro_L;    
    double radio_actual, radio_anterior;    
    
    double direccion_x, direccion_y, direccion_L; 
    
    double jacobiano[3][3];      
    double jacobiano_inverso[3][3];       
    double error_funcion[3][1];          
    double paso_predicho[3][1];          
    double paso_corregido[3][1];        
    
    double energia_acumulada_meta;                
    double energia_acumulada_inicio;              
    int contador_pasos;                 
    int total_iteraciones_newton;                
    double determinante_anterior;             
} EstadoRobot;

void invertir_matriz_jacobiana(double J[3][3], double J_inv[3][3]) {
    double cofactor_00 = J[1][1]*J[2][2] - J[1][2]*J[2][1];
    double cofactor_01 = J[1][0]*J[2][2] - J[1][2]*J[2][0];
    double cofactor_02 = J[1][0]*J[2][1] - J[1][1]*J[2][0];

    double determinante = J[0][0]*cofactor_00 - J[0][1]*cofactor_01 + J[0][2]*cofactor_02;
    double inv_determinante = (fabs(determinante) < LIMITE_CERO_MATRIZ) ? 0.0 : 1.0 / determinante;

    J_inv[0][0] =  cofactor_00 * inv_determinante;
    J_inv[1][0] = -cofactor_01 * inv_determinante;
    J_inv[2][0] =  cofactor_02 * inv_determinante;
    J_inv[0][1] = -(J[0][1]*J[2][2] - J[0][2]*J[2][1]) * inv_determinante;
    J_inv[1][1] =  (J[0][0]*J[2][2] - J[0][2]*J[2][0]) * inv_determinante;
    J_inv[2][1] = -(J[0][0]*J[2][1] - J[0][1]*J[2][0]) * inv_determinante;
    J_inv[0][2] =  (J[0][1]*J[1][2] - J[0][2]*J[1][1]) * inv_determinante;
    J_inv[1][2] = -(J[0][0]*J[1][2] - J[0][2]*J[1][0]) * inv_determinante;
    J_inv[2][2] =  (J[0][0]*J[1][1] - J[0][1]*J[1][0]) * inv_determinante;
}

void evaluar_entorno_y_errores(EntornoNavegacion *entorno, EstadoRobot *robot, int calcular_errores_f) {
    double sumatoria_repulsion_x = 0.0, sumatoria_repulsion_y = 0.0, sumatoria_energia_W = 0.0;
    
    for(int k = 0; k < entorno->num_obstaculos; k++) {
        double delta_x = robot->x - entorno->matriz_obstaculos[k][0];
        double delta_y = robot->y - entorno->matriz_obstaculos[k][1];
        
        double distancia_cuadrada = (delta_x * delta_x) + (delta_y * delta_y);
        double distancia_superficie = distancia_cuadrada - entorno->matriz_obstaculos[k][2]; 
        double raiz_distancia = sqrt((distancia_superficie * distancia_superficie) + EPSILON_CUADRADO); 
        
        double denominador_repulsion = distancia_superficie + (raiz_distancia - EPSILON_SEGURIDAD); 
        if(fabs(denominador_repulsion) < LIMITE_CERO_MATRIZ) denominador_repulsion = LIMITE_CERO_MATRIZ;

        double derivada_distancia = 1.0 + (distancia_superficie / raiz_distancia); 
        
        // El peso neto YA VIENE CON SIGNO y atenuación desde Python
        double peso_neto = entorno->matriz_obstaculos[k][3]; 
        
        double factor_fuerza = -(peso_neto) / (denominador_repulsion * denominador_repulsion) * derivada_distancia;

        sumatoria_repulsion_x += (factor_fuerza * (2.0 * delta_x)); 
        sumatoria_repulsion_y += (factor_fuerza * (2.0 * delta_y));
        
        if(calcular_errores_f) {
            sumatoria_energia_W += (peso_neto) / denominador_repulsion;
        }
    }

    robot->jacobiano[0][0] = -entorno->pendiente_inicial; 
    robot->jacobiano[0][1] = -1.0; 
    robot->jacobiano[0][2] = -entorno->inicio_y - entorno->pendiente_inicial * entorno->inicio_x + entorno->pendiente_inicial * entorno->meta_x + entorno->meta_y;

    robot->jacobiano[1][0] = -entorno->pendiente_final + sumatoria_repulsion_x; 
    robot->jacobiano[1][1] = -1.0 + sumatoria_repulsion_y;
    robot->jacobiano[1][2] = -entorno->inicio_y - entorno->pendiente_final*entorno->inicio_x + entorno->meta_y + entorno->pendiente_final*entorno->meta_x + robot->energia_acumulada_inicio - robot->energia_acumulada_meta;
    
    robot->jacobiano[2][0] = 2.0 * (robot->x - robot->centro_x); 
    robot->jacobiano[2][1] = 2.0 * (robot->y - robot->centro_y);
    robot->jacobiano[2][2] = 2.0 * (robot->L - robot->centro_L);

    if (calcular_errores_f) {
        double constante_H1 = -entorno->inicio_y - entorno->pendiente_inicial*entorno->inicio_x + entorno->pendiente_inicial*entorno->meta_x + entorno->meta_y;
        double constante_H2 = -entorno->inicio_y - entorno->pendiente_final*entorno->inicio_x + (entorno->meta_y + entorno->pendiente_final*entorno->meta_x) + robot->energia_acumulada_inicio - robot->energia_acumulada_meta;
        
        robot->error_funcion[0][0] = -robot->y - entorno->pendiente_inicial*robot->x + entorno->pendiente_inicial*entorno->meta_x + entorno->meta_y - (1.0 - robot->L)*constante_H1;
        robot->error_funcion[1][0] = (-robot->y - entorno->pendiente_final*robot->x + (entorno->meta_y + entorno->pendiente_final*entorno->meta_x) + sumatoria_energia_W - robot->energia_acumulada_meta) - (1.0 - robot->L)*constante_H2;
        
        double distancia_centro_x = robot->x - robot->centro_x;
        double distancia_centro_y = robot->y - robot->centro_y;
        double distancia_centro_L = robot->L - robot->centro_L;
        robot->error_funcion[2][0] = (distancia_centro_x * distancia_centro_x) + (distancia_centro_y * distancia_centro_y) + (distancia_centro_L * distancia_centro_L) - (robot->radio_actual * robot->radio_actual);
    }
}

void leer_archivos_configuracion(EntornoNavegacion *entorno) {
    FILE *archivo_pendientes = fopen("pendientes.txt", "r");
    if (!archivo_pendientes) { printf("Error: 'pendientes.txt'\n"); exit(1); }

    double temp_m1, temp_m2;
    fscanf(archivo_pendientes, "%lf", &entorno->radio_esfera_paso);
    fscanf(archivo_pendientes, "%lf", &temp_m1);
    fscanf(archivo_pendientes, "%lf", &temp_m2);
    
    entorno->pendiente_inicial = 1.0;
    entorno->pendiente_final = 4.0;
    
    fscanf(archivo_pendientes, "%d", &entorno->max_pasos_permitidos);
    if(entorno->max_pasos_permitidos > MAX_PASOS_HISTORIAL) entorno->max_pasos_permitidos = MAX_PASOS_HISTORIAL;
    
    fscanf(archivo_pendientes, "%lf", &entorno->inicio_x);
    fscanf(archivo_pendientes, "%lf", &entorno->inicio_y);
    fscanf(archivo_pendientes, "%lf", &entorno->meta_x);
    fscanf(archivo_pendientes, "%lf", &entorno->meta_y);
    fclose(archivo_pendientes);

    FILE *archivo_obstaculos = fopen("obstaculos.txt", "r");
    if (!archivo_obstaculos) { printf("Error: 'obstaculos.txt'\n"); exit(1); }
    
    entorno->num_obstaculos = 0;
    double v0, v1, v2, v3;
    
    // Leemos x, y, r, p_neto (Python ya lo calculó y clusterizó)
    while (fscanf(archivo_obstaculos, "%lf %lf %lf %lf", &v0, &v1, &v2, &v3) == 4) { 
        if (entorno->num_obstaculos < MAX_OBSTACULOS) {
            entorno->matriz_obstaculos[entorno->num_obstaculos][0] = v0;            
            entorno->matriz_obstaculos[entorno->num_obstaculos][1] = v1;            
            entorno->matriz_obstaculos[entorno->num_obstaculos][2] = v2 * v2;       
            entorno->matriz_obstaculos[entorno->num_obstaculos][3] = v3; 
            entorno->num_obstaculos++;
        }
    }
    fclose(archivo_obstaculos);
}

void planificar_ruta_homotopica(EntornoNavegacion *entorno) {
    EstadoRobot robot = {0}; 
    
    robot.determinante_anterior = 1.0; 
    robot.contador_pasos = 1;
    robot.radio_actual = entorno->radio_esfera_paso; 
    robot.radio_anterior = entorno->radio_esfera_paso;

    robot.x = entorno->inicio_x; 
    robot.y = entorno->inicio_y;
    robot.x_anterior = entorno->inicio_x; 
    robot.y_anterior = entorno->inicio_y;
    
    double termino_L = (entorno->meta_y + (entorno->pendiente_inicial * (entorno->meta_x - entorno->inicio_x)) - entorno->inicio_y);
    if (fabs(termino_L) < 1e-9) termino_L = 1e-9;

    robot.L_anterior = (robot.y_anterior + (entorno->pendiente_inicial * robot.x_anterior) - (entorno->inicio_y + entorno->pendiente_inicial * entorno->inicio_x)) / termino_L; 
    robot.L = robot.L_anterior;
    
    robot.centro_x = robot.x_anterior; 
    robot.centro_y = robot.y_anterior; 
    robot.centro_L = robot.L_anterior; 
    
    int signo_direccion = (entorno->pendiente_inicial > entorno->pendiente_final) ? -1 : 1; 
    double ultimos_centros[3][3] = {0}; 
    int meta_alcanzada = 0;

    entorno->historial_ruta[0][0] = robot.x_anterior;
    entorno->historial_ruta[0][1] = robot.y_anterior;
    entorno->historial_ruta[0][2] = robot.L_anterior;

    for(int k = 0; k < entorno->num_obstaculos; k++) {
        double peso_neto = entorno->matriz_obstaculos[k][3];
        double radio_sq = entorno->matriz_obstaculos[k][2];

        double dx_meta = entorno->meta_x - entorno->matriz_obstaculos[k][0];
        double dy_meta = entorno->meta_y - entorno->matriz_obstaculos[k][1];
        double dist_meta = (dx_meta * dx_meta) + (dy_meta * dy_meta) - radio_sq;
        double raiz_meta = sqrt((dist_meta * dist_meta) + EPSILON_CUADRADO);
        double denominador_meta = dist_meta + (raiz_meta - EPSILON_SEGURIDAD);
        if (fabs(denominador_meta) < LIMITE_CERO_MATRIZ) denominador_meta = LIMITE_CERO_MATRIZ;
        robot.energia_acumulada_meta += (peso_neto) / denominador_meta; 

        double dx_inicio = entorno->inicio_x - entorno->matriz_obstaculos[k][0];
        double dy_inicio = entorno->inicio_y - entorno->matriz_obstaculos[k][1];
        double dist_inicio = (dx_inicio * dx_inicio) + (dy_inicio * dy_inicio) - radio_sq;
        double raiz_inicio = sqrt((dist_inicio * dist_inicio) + EPSILON_CUADRADO);
        double denominador_inicio = dist_inicio + (raiz_inicio - EPSILON_SEGURIDAD);
        if (fabs(denominador_inicio) < LIMITE_CERO_MATRIZ) denominador_inicio = LIMITE_CERO_MATRIZ;
        robot.energia_acumulada_inicio += (peso_neto) / denominador_inicio;
    }

    evaluar_entorno_y_errores(entorno, &robot, FALSO);
    
    double vector_tangente_x = robot.jacobiano[0][2]*robot.jacobiano[1][1] - robot.jacobiano[1][2]*robot.jacobiano[0][1];
    double vector_tangente_y = robot.jacobiano[1][2]*robot.jacobiano[0][0] - robot.jacobiano[0][2]*robot.jacobiano[1][0];
    double vector_tangente_L = -(robot.jacobiano[0][0]*robot.jacobiano[1][1] - robot.jacobiano[0][1]*robot.jacobiano[1][0]);
    
    double norma_tangente = sqrt((vector_tangente_x*vector_tangente_x) + (vector_tangente_y*vector_tangente_y) + (vector_tangente_L*vector_tangente_L));
    if (norma_tangente < LIMITE_CERO_NORMA) norma_tangente = LIMITE_CERO_NORMA;
    
    robot.direccion_x = vector_tangente_x / norma_tangente; 
    robot.direccion_y = vector_tangente_y / norma_tangente; 
    robot.direccion_L = vector_tangente_L / norma_tangente;

    robot.paso_predicho[0][0] = robot.x_anterior + signo_direccion*(entorno->radio_esfera_paso * robot.direccion_x);
    robot.paso_predicho[1][0] = robot.y_anterior + signo_direccion*(entorno->radio_esfera_paso * robot.direccion_y);
    robot.paso_predicho[2][0] = robot.L_anterior + signo_direccion*(entorno->radio_esfera_paso * robot.direccion_L);
    robot.determinante_anterior = vector_tangente_L; 

    while ((robot.contador_pasos < entorno->max_pasos_permitidos - 2) && (robot.L > -0.5) && (meta_alcanzada == 0)) {
        
        double respaldo_x = robot.x_anterior; 
        double respaldo_y = robot.y_anterior; 
        double respaldo_L = robot.L_anterior;

        robot.x = robot.paso_predicho[0][0]; 
        robot.y = robot.paso_predicho[1][0]; 
        robot.L = robot.paso_predicho[2][0];
        
        memcpy(&ultimos_centros[0][0], &ultimos_centros[1][0], 3 * sizeof(double));
        memcpy(&ultimos_centros[1][0], &ultimos_centros[2][0], 3 * sizeof(double));
        ultimos_centros[2][0] = robot.centro_x; 
        ultimos_centros[2][1] = robot.centro_y; 
        ultimos_centros[2][2] = robot.centro_L;

        entorno->historial_ruta[robot.contador_pasos][0] = robot.x_anterior; 
        entorno->historial_ruta[robot.contador_pasos][1] = robot.y_anterior; 
        entorno->historial_ruta[robot.contador_pasos][2] = robot.L_anterior;

        int iteraciones_actuales = 0; 
        double error_cuadrado = 1.0;
        
        while (iteraciones_actuales < MAX_ITERACIONES_NEWTON) {
            evaluar_entorno_y_errores(entorno, &robot, VERDADERO);
            error_cuadrado = (robot.error_funcion[0][0]*robot.error_funcion[0][0]) + 
                             (robot.error_funcion[1][0]*robot.error_funcion[1][0]) + 
                             (robot.error_funcion[2][0]*robot.error_funcion[2][0]);
                             
            if (error_cuadrado <= TOLERANCIA_ERROR_F_CUADRADO) break;

            robot.jacobiano[0][0] += FACTOR_AMORTIGUACION;
            robot.jacobiano[1][1] += FACTOR_AMORTIGUACION;
            robot.jacobiano[2][2] += FACTOR_AMORTIGUACION;

            invertir_matriz_jacobiana(robot.jacobiano, robot.jacobiano_inverso);
            
            double ajuste_x = (robot.jacobiano_inverso[0][0]*robot.error_funcion[0][0] + robot.jacobiano_inverso[0][1]*robot.error_funcion[1][0] + robot.jacobiano_inverso[0][2]*robot.error_funcion[2][0]);
            double ajuste_y = (robot.jacobiano_inverso[1][0]*robot.error_funcion[0][0] + robot.jacobiano_inverso[1][1]*robot.error_funcion[1][0] + robot.jacobiano_inverso[1][2]*robot.error_funcion[2][0]);
            double ajuste_L = (robot.jacobiano_inverso[2][0]*robot.error_funcion[0][0] + robot.jacobiano_inverso[2][1]*robot.error_funcion[1][0] + robot.jacobiano_inverso[2][2]*robot.error_funcion[2][0]);
            
            robot.x -= ajuste_x; 
            robot.y -= ajuste_y; 
            robot.L -= ajuste_L;
            
            robot.x_anterior = robot.x; 
            robot.y_anterior = robot.y; 
            robot.L_anterior = robot.L;
            
            iteraciones_actuales++;
        }

        if (error_cuadrado > TOLERANCIA_ERROR_F_CUADRADO) {
            robot.radio_actual /= 2.0;
            if (robot.radio_actual < 1e-6) {
                break; 
            }
            robot.x_anterior = respaldo_x; 
            robot.y_anterior = respaldo_y; 
            robot.L_anterior = respaldo_L;
            
            robot.paso_predicho[0][0] = robot.centro_x + signo_direccion * (robot.radio_actual * robot.direccion_x);
            robot.paso_predicho[1][0] = robot.centro_y + signo_direccion * (robot.radio_actual * robot.direccion_y);
            robot.paso_predicho[2][0] = robot.centro_L + signo_direccion * (robot.radio_actual * robot.direccion_L);
            continue; 
        }

        robot.total_iteraciones_newton += iteraciones_actuales;

        if (robot.contador_pasos > 3) {
            int robot_atascado = 0, intentos_rotacion = 1, direccion_giro = 1;
            double gradiente_anterior[3], gradiente_actual[3]; 
            
            gradiente_anterior[0] = 2.0 * (entorno->historial_ruta[robot.contador_pasos-1][0] - ultimos_centros[2][0]);
            gradiente_anterior[1] = 2.0 * (entorno->historial_ruta[robot.contador_pasos-1][1] - ultimos_centros[2][1]);
            gradiente_anterior[2] = 2.0 * (entorno->historial_ruta[robot.contador_pasos-1][2] - ultimos_centros[2][2]);
            double norma_anterior = sqrt(gradiente_anterior[0]*gradiente_anterior[0] + gradiente_anterior[1]*gradiente_anterior[1] + gradiente_anterior[2]*gradiente_anterior[2]);
            
            robot.x = robot.x_anterior; 
            robot.y = robot.y_anterior; 
            robot.L = robot.L_anterior;
            
            gradiente_actual[0] = 2.0 * (robot.x - ultimos_centros[2][0]); 
            gradiente_actual[1] = 2.0 * (robot.y - ultimos_centros[2][1]); 
            gradiente_actual[2] = 2.0 * (robot.L - ultimos_centros[2][2]);
            double norma_actual = sqrt(gradiente_actual[0]*gradiente_actual[0] + gradiente_actual[1]*gradiente_actual[1] + gradiente_actual[2]*gradiente_actual[2]);

            if (norma_anterior > LIMITE_CERO_NORMA && norma_actual > LIMITE_CERO_NORMA) {
                double similitud = (gradiente_anterior[0]*gradiente_actual[0] + gradiente_anterior[1]*gradiente_actual[1] + gradiente_anterior[2]*gradiente_actual[2]) / (norma_anterior * norma_actual);
                if (similitud > UMBRAL_RETROCESO_ANGULAR) robot_atascado = 1; 
            }

            while ((intentos_rotacion < 3) && (robot_atascado == 1)) {
                double delta_x_centro = ultimos_centros[2][0] - entorno->historial_ruta[robot.contador_pasos-1][0];
                double delta_y_centro = ultimos_centros[2][1] - entorno->historial_ruta[robot.contador_pasos-1][1];
                double norma_centro = sqrt(delta_x_centro*delta_x_centro + delta_y_centro*delta_y_centro);
                if(norma_centro < LIMITE_CERO_NORMA) norma_centro = LIMITE_CERO_NORMA;
                
                double angulo_base = acos(LIMITAR_ACOS(delta_x_centro/norma_centro)) * (180.0/PI);
                
                evaluar_entorno_y_errores(entorno, &robot, FALSO); 
                double tangente_x_aux = signo_direccion*(robot.radio_actual*robot.direccion_x); 
                
                if (intentos_rotacion == 1) {
                    double angulo_tangente = acos(LIMITAR_ACOS(tangente_x_aux)) * (180.0/PI);
                    direccion_giro = (angulo_tangente > angulo_base) ? 1 : -1;
                } else {
                    direccion_giro *= -1; 
                }

                double angulo_escape = (PI / 4.0) * direccion_giro; 
                double cos_angulo = cos(angulo_escape);
                double sin_angulo = sin(angulo_escape);
                
                double punto_referencia[2] = {robot.paso_predicho[0][0] - ultimos_centros[2][0], robot.paso_predicho[1][0] - ultimos_centros[2][1]};
                double punto_rotado[2] = {cos_angulo*punto_referencia[0] - sin_angulo*punto_referencia[1], sin_angulo*punto_referencia[0] + cos_angulo*punto_referencia[1]};
                
                robot.paso_corregido[0][0] = punto_rotado[0] + ultimos_centros[2][0]; 
                robot.paso_corregido[1][0] = punto_rotado[1] + ultimos_centros[2][1];
                robot.paso_corregido[2][0] = (robot.paso_corregido[1][0] + (entorno->pendiente_inicial*robot.paso_corregido[0][0]) - (entorno->inicio_y+entorno->pendiente_inicial*entorno->inicio_x)) / termino_L;

                robot.x = robot.paso_corregido[0][0]; 
                robot.y = robot.paso_corregido[1][0]; 
                robot.L = robot.paso_corregido[2][0];
                
                iteraciones_actuales = 0; 
                error_cuadrado = 1.0;
                
                while (iteraciones_actuales < MAX_ITERACIONES_NEWTON) {
                    evaluar_entorno_y_errores(entorno, &robot, VERDADERO);
                    error_cuadrado = (robot.error_funcion[0][0]*robot.error_funcion[0][0]) + (robot.error_funcion[1][0]*robot.error_funcion[1][0]) + (robot.error_funcion[2][0]*robot.error_funcion[2][0]);
                    if (error_cuadrado <= TOLERANCIA_ERROR_F_CUADRADO) break;

                    robot.jacobiano[0][0] += FACTOR_AMORTIGUACION;
                    robot.jacobiano[1][1] += FACTOR_AMORTIGUACION;
                    robot.jacobiano[2][2] += FACTOR_AMORTIGUACION;

                    invertir_matriz_jacobiana(robot.jacobiano, robot.jacobiano_inverso);
                    double ajuste_rot_x = (robot.jacobiano_inverso[0][0]*robot.error_funcion[0][0] + robot.jacobiano_inverso[0][1]*robot.error_funcion[1][0] + robot.jacobiano_inverso[0][2]*robot.error_funcion[2][0]);
                    double ajuste_rot_y = (robot.jacobiano_inverso[1][0]*robot.error_funcion[0][0] + robot.jacobiano_inverso[1][1]*robot.error_funcion[1][0] + robot.jacobiano_inverso[1][2]*robot.error_funcion[2][0]);
                    double ajuste_rot_L = (robot.jacobiano_inverso[2][0]*robot.error_funcion[0][0] + robot.jacobiano_inverso[2][1]*robot.error_funcion[1][0] + robot.jacobiano_inverso[2][2]*robot.error_funcion[2][0]);
                    
                    robot.x -= ajuste_rot_x; 
                    robot.y -= ajuste_rot_y; 
                    robot.L -= ajuste_rot_L; 
                    
                    robot.x_anterior = robot.x; 
                    robot.y_anterior = robot.y; 
                    robot.L_anterior = robot.L;
                    iteraciones_actuales++;
                }
                robot.total_iteraciones_newton += iteraciones_actuales;

                if (error_cuadrado <= TOLERANCIA_ERROR_F_CUADRADO) {
                    double gradiente_escape[3] = {2.0*(robot.x - ultimos_centros[2][0]), 2.0*(robot.y - ultimos_centros[2][1]), 2.0*(robot.L - ultimos_centros[2][2])};
                    double norma_escape = sqrt(gradiente_escape[0]*gradiente_escape[0] + gradiente_escape[1]*gradiente_escape[1] + gradiente_escape[2]*gradiente_escape[2]);
                    
                    double similitud_escape = -1.0;
                    if (norma_anterior > LIMITE_CERO_NORMA && norma_escape > LIMITE_CERO_NORMA) {
                        similitud_escape = (gradiente_anterior[0]*gradiente_escape[0] + gradiente_anterior[1]*gradiente_escape[1] + gradiente_anterior[2]*gradiente_escape[2]) / (norma_anterior * norma_escape);
                    }
                    if (similitud_escape < UMBRAL_SALIDA_RETROCESO) { 
                        robot_atascado = 0; 
                        break;
                    }
                }
                intentos_rotacion++;
            }
        }

        evaluar_entorno_y_errores(entorno, &robot, FALSO);
        
        double sub_determinante = -1.0 * (robot.jacobiano[0][0]*robot.jacobiano[1][1] - robot.jacobiano[0][1]*robot.jacobiano[1][0]);
        double denominador_tangente = robot.jacobiano[0][0]*robot.jacobiano[1][1] - robot.jacobiano[0][1]*robot.jacobiano[1][0];
        if(fabs(denominador_tangente) < LIMITE_CERO_NORMA) denominador_tangente = LIMITE_CERO_NORMA; 

        double componente_tangente_x = (-sub_determinante*robot.jacobiano[0][2])*(robot.jacobiano[1][1]/denominador_tangente) + (-sub_determinante*robot.jacobiano[1][2])*(-robot.jacobiano[0][1]/denominador_tangente);
        double componente_tangente_y = (-sub_determinante*robot.jacobiano[0][2])*(-robot.jacobiano[1][0]/denominador_tangente) + (-sub_determinante*robot.jacobiano[1][2])*(robot.jacobiano[0][0]/denominador_tangente);
        double componente_tangente_L = sub_determinante;
        
        double norma_nueva_direccion = sqrt((componente_tangente_x*componente_tangente_x) + (componente_tangente_y*componente_tangente_y) + (componente_tangente_L*componente_tangente_L));
        if(norma_nueva_direccion < LIMITE_CERO_NORMA) norma_nueva_direccion = LIMITE_CERO_NORMA;
        
        robot.direccion_x = componente_tangente_x / norma_nueva_direccion; 
        robot.direccion_y = componente_tangente_y / norma_nueva_direccion; 
        robot.direccion_L = componente_tangente_L / norma_nueva_direccion;
        
        double marcador_estabilidad = sub_determinante;
        double factor_exponencial = exp(-fabs(robot.determinante_anterior) / sqrt((marcador_estabilidad*marcador_estabilidad) + LIMITE_CERO_NORMA));
        double radio_paso_sugerido = entorno->radio_esfera_paso * (1.0 + factor_exponencial); 
        
        if (radio_paso_sugerido > 1.2 * robot.radio_actual) {
            robot.radio_actual = 1.2 * robot.radio_actual;
            if (robot.radio_actual > entorno->radio_esfera_paso * 2.0) {
                robot.radio_actual = entorno->radio_esfera_paso * 2.0;
            }
        } else {
            robot.radio_actual = radio_paso_sugerido;
        }
        
        robot.determinante_anterior = marcador_estabilidad;
        
        if ((robot.radio_anterior - robot.radio_actual) > (robot.radio_anterior / 3.0)) {
            robot.radio_actual = (robot.radio_anterior / 2.0) + entorno->radio_esfera_paso;
        }
        
        double vector_avance_x = robot.x_anterior - entorno->historial_ruta[robot.contador_pasos][0];
        double vector_avance_y = robot.y_anterior - entorno->historial_ruta[robot.contador_pasos][1];
        double vector_avance_L = robot.L_anterior - entorno->historial_ruta[robot.contador_pasos][2];
        
        double distancia_avance = sqrt((vector_avance_x*vector_avance_x) + (vector_avance_y*vector_avance_y) + (vector_avance_L*vector_avance_L));
        if(fabs(distancia_avance) < LIMITE_CERO_NORMA) distancia_avance = LIMITE_CERO_NORMA;

        double factor_escala_centro = (robot.radio_actual - robot.radio_anterior) / distancia_avance;
        
        robot.centro_x = robot.x_anterior + factor_escala_centro * vector_avance_x;
        robot.centro_y = robot.y_anterior + factor_escala_centro * vector_avance_y;
        robot.centro_L = robot.L_anterior + factor_escala_centro * vector_avance_L;
        
        robot.paso_predicho[0][0] = robot.centro_x + signo_direccion*(robot.radio_actual*robot.direccion_x);
        robot.paso_predicho[1][0] = robot.centro_y + signo_direccion*(robot.radio_actual*robot.direccion_y);
        robot.paso_predicho[2][0] = robot.centro_L + signo_direccion*(robot.radio_actual*robot.direccion_L);

        double distancia_cuadrada_meta = (robot.x_anterior - entorno->meta_x)*(robot.x_anterior - entorno->meta_x) + 
                                         (robot.y_anterior - entorno->meta_y)*(robot.y_anterior - entorno->meta_y);
                                         
        if (robot.L_anterior >= 1.0 && distancia_cuadrada_meta <= (robot.radio_actual * robot.radio_actual)) {
            meta_alcanzada = 1;
        }

        if (robot.contador_pasos > VENTANA_ESTANCAMIENTO) {
            int indice_pasado = robot.contador_pasos - VENTANA_ESTANCAMIENTO;
            double dx_estancamiento = robot.x_anterior - entorno->historial_ruta[indice_pasado][0];
            double dy_estancamiento = robot.y_anterior - entorno->historial_ruta[indice_pasado][1];
            
            double progreso_cuadrado = (dx_estancamiento * dx_estancamiento) + (dy_estancamiento * dy_estancamiento);
            
            if (progreso_cuadrado < (MIN_DISTANCIA_PROGRESO * MIN_DISTANCIA_PROGRESO)) {
                break; 
            }
        }

        robot.radio_anterior = robot.radio_actual;
        robot.contador_pasos++;
    }

    entorno->historial_ruta[robot.contador_pasos][0] = robot.x_anterior;
    entorno->historial_ruta[robot.contador_pasos][1] = robot.y_anterior;
    entorno->historial_ruta[robot.contador_pasos][2] = robot.L_anterior;
    entorno->pasos_totales_dados = robot.contador_pasos + 1; 
}  

void exportar_ruta_calculada(EntornoNavegacion *entorno) {
    FILE *archivo_salida = fopen("tray.txt", "w");
    if (!archivo_salida) return;
    for (int i = 0; i < entorno->pasos_totales_dados; i++){
        fprintf(archivo_salida, "%f \t %f \n", entorno->historial_ruta[i][0], entorno->historial_ruta[i][1]);
    }
    fclose(archivo_salida);
}

int main() {
    EntornoNavegacion entorno_actual = {0};

    leer_archivos_configuracion(&entorno_actual);
    planificar_ruta_homotopica(&entorno_actual); 
    exportar_ruta_calculada(&entorno_actual);

    return 0;
}