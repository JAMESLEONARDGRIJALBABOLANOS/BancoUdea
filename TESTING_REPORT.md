# Reporte Completo de Testing — BancoUdea
## Una guía de profesor a estudiante

---

## Tabla de Contenidos

1. [Fundamentos — ¿Por qué hacemos pruebas?](#1-fundamentos)
2. [El Proyecto BancoUdea](#2-el-proyecto-bancoudea)
3. [Pruebas Unitarias — JUnit 5 + Mockito](#3-pruebas-unitarias--junit-5--mockito)
4. [Cucumber + Gherkin — Pruebas BDD](#4-cucumber--gherkin--pruebas-bdd)
5. [Karate — Pruebas de API REST](#5-karate--pruebas-de-api-rest)
6. [Análisis de Calidad — SonarCloud](#6-análisis-de-calidad--sonarcloud)
7. [Comparativa Final](#7-comparativa-final)
8. [Lección Final](#8-lección-final)

---

## 1. Fundamentos

### ¿Por qué hacemos pruebas?

Imagina que construyes un puente. Antes de que los carros pasen por él, los ingenieros hacen pruebas: prueban cada viga por separado, luego prueban cómo se conectan las vigas, y finalmente prueban el puente completo con carga real. Si solo prueban al final y el puente falla, no saben exactamente qué parte falló.

En software pasa exactamente lo mismo. Tenemos tres tipos de pruebas que corresponden a esos tres momentos:

```
                    /\
                   /  \
                  / E2E \          ← Karate: prueba todo junto (el puente completo)
                 /--------\
                /Integración\      ← Cucumber: prueba partes conectadas (vigas ensambladas)
               /--------------\
              /   Unitarias    \   ← JUnit + Mockito: prueba cada pieza sola (cada viga)
             /------------------\
```

Esta figura se llama **Pirámide de Testing**. La base es más ancha porque hay más pruebas unitarias (son baratas y rápidas), y la punta es más delgada porque las pruebas E2E son pocas pero cubren mucho.

---

## 2. El Proyecto BancoUdea

### 2.1 ¿Qué es BancoUdea?

Es una API REST (un servicio web) que simula operaciones bancarias básicas. No tiene pantallas — solo recibe y responde peticiones HTTP.

### 2.2 Arquitectura del sistema

```
Cliente (Postman, navegador, app)
            |
            | HTTP Request
            v
    ┌─────────────────┐
    │   Controller    │  ← Recibe la petición y la dirige
    └────────┬────────┘
             |
    ┌────────▼────────┐
    │    Service      │  ← Aquí vive la lógica de negocio
    └────────┬────────┘
             |
    ┌────────▼────────┐
    │   Repository    │  ← Habla con la base de datos
    └────────┬────────┘
             |
    ┌────────▼────────┐
    │   MySQL (BD)    │  ← Guarda los datos permanentemente
    └─────────────────┘
```

### 2.3 Las dos entidades del sistema

**Customer (Cliente bancario)**

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | Long | Número único autogenerado |
| `accountNumber` | String | Número de cuenta — ÚNICO |
| `firstName` | String | Nombre del cliente |
| `lastName` | String | Apellido del cliente |
| `balance` | Double | Saldo disponible |

**Transaction (Transacción)**

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | Long | Número único autogenerado |
| `senderAccountNumber` | String | Cuenta que envía el dinero |
| `receiverAccountNumber` | String | Cuenta que recibe el dinero |
| `amount` | Double | Monto transferido |
| `timestamp` | LocalDateTime | Fecha y hora de la operación |

### 2.4 Endpoints disponibles

```
Clientes:
GET    /api/customers         → listar todos
GET    /api/customers/{id}    → buscar por ID
POST   /api/customers         → crear nuevo
PUT    /api/customers/{id}    → actualizar
DELETE /api/customers/{id}    → eliminar

Transacciones:
POST   /api/transactions/transfer          → transferir dinero
GET    /api/transactions/{accountNumber}   → ver historial
```

### 2.5 Reglas de negocio críticas

1. No se puede transferir si el saldo es insuficiente
2. No se puede transferir a/desde una cuenta inexistente
3. Los números de cuenta no pueden ser nulos en una transferencia
4. Al crear un cliente, el balance no puede ser nulo
5. El número de cuenta debe ser único en el sistema

---

## 3. Pruebas Unitarias — JUnit 5 + Mockito

### 3.1 ¿Qué es una prueba unitaria?

Una prueba unitaria verifica **una sola pieza del código de forma completamente aislada**. La palabra clave es "aislada" — no queremos que la prueba dependa de la base de datos, del servidor, ni de ningún componente externo.

> **Analogía:** Imagina que fabricas autos. Una prueba unitaria sería probar el motor en el laboratorio, desconectado del auto, sin ruedas, sin carrocería. Solo el motor. Si el motor falla, sabes exactamente dónde está el problema.

### 3.2 ¿Qué es Mockito y por qué lo necesitamos?

El `CustomerService` necesita al `CustomerRepository`, y el `CustomerRepository` necesita a MySQL. Pero en pruebas unitarias **no queremos conectarnos a MySQL**.

**Mockito** crea objetos "falsos" (mocks) que simulan el comportamiento de las dependencias reales:

```
SIN MOCKITO (lo que queremos EVITAR):
CustomerService → CustomerRepository → MySQL ← PROBLEMA

CON MOCKITO (lo que hacemos):
CustomerService → CustomerRepository (FALSO/MOCK) ← Perfecto
```

### 3.3 Vocabulario fundamental de Mockito

```java
@Mock           // Crea un objeto falso de esa clase
@InjectMocks    // Crea el objeto real e inyecta los mocks automáticamente
@ExtendWith(MockitoExtension.class) // Activa Mockito en la clase de test

when(...).thenReturn(...)   // "Cuando llamen esto, responde esto"
verify(mock).metodo()       // "Verifica que este método fue llamado"
assertThrows(...)           // "Verifica que se lanza esta excepción"
```

### 3.4 Estructura de una prueba unitaria — Patrón AAA

Cada prueba unitaria bien escrita sigue el patrón **Arrange - Act - Assert**:

```java
@Test
void transferMoney_realizaTransferenciaCorrectamente() {

    // ARRANGE (Preparar) — configura el escenario
    sender = new Customer(1L, "ACC-001", "Juan", "Perez", 1000.0);
    when(customerRepository.findByAccountNumber("ACC-001"))
        .thenReturn(Optional.of(sender));

    // ACT (Actuar) — ejecuta lo que quieres probar
    TransactionDTO result = transactionService.transferMoney(transactionDTO);

    // ASSERT (Verificar) — comprueba que el resultado es correcto
    assertEquals(800.0, sender.getBalance());
    verify(transactionRepository).save(any(Transaction.class));
}
```

### 3.5 CustomerServiceTest — Casos de prueba

**Configuración inicial:**

```java
@ExtendWith(MockitoExtension.class)
class CustomerServiceTest {

    @Mock
    private CustomerRepository customerRepository; // BD simulada

    @Mock
    private CustomerMapper customerMapper;         // Mapper simulado

    @InjectMocks
    private CustomerService customerService;       // El que realmente probamos

    @BeforeEach
    void setUp() {
        // Se ejecuta antes de CADA test — datos frescos para cada prueba
        customer = new Customer(1L, "ACC-001", "Juan", "Perez", 1000.0);
        customerDTO = new CustomerDTO(1L, "Juan", "Perez", "ACC-001", 1000.0);
    }
}
```

**Tabla de casos:**

| # | Método | Escenario | Resultado esperado |
|---|---|---|---|
| 1 | `getAllCustomer` | Existen clientes en BD | Retorna lista de DTOs mapeados |
| 2 | `getAllCustomer` | No hay clientes | Retorna lista vacía |
| 3 | `getCustomerById` | ID existe | Retorna CustomerDTO correcto |
| 4 | `getCustomerById` | ID no existe | Lanza `RuntimeException("Cliente no encontrado")` |
| 5 | `createCustomer` | DTO válido | Guarda y retorna DTO con datos correctos |
| 6 | `updateCustomer` | Campos no nulos | Actualiza solo los campos enviados |
| 7 | `updateCustomer` | Cliente no existe | Lanza `RuntimeException("Cliente no encontrado")` |
| 8 | `deleteCustomer` | Cliente existe | Elimina correctamente sin excepción |
| 9 | `deleteCustomer` | Cliente no existe | Lanza excepción, nunca llama `deleteById` |

**Ejemplo destacado — Test de excepción:**

```java
@Test
void getCustomerById_lanzaExcepcionCuandoNoExiste() {
    when(customerRepository.findById(99L)).thenReturn(Optional.empty());

    RuntimeException ex = assertThrows(RuntimeException.class,
            () -> customerService.getCustomerById(99L));

    assertEquals("Cliente no encontrado", ex.getMessage());
    // Si el mensaje cambia en el código, este test falla y nos avisa
}
```

**Ejemplo destacado — Verificación negativa:**

```java
@Test
void deleteCustomer_lanzaExcepcionSiClienteNoExiste() {
    when(customerRepository.existsById(99L)).thenReturn(false);

    assertThrows(RuntimeException.class,
            () -> customerService.deleteCustomer(99L));

    // Garantiza que deleteById NUNCA se llama si el cliente no existe
    verify(customerRepository, never()).deleteById(any());
}
```

### 3.6 TransactionServiceTest — Casos de prueba

| # | Método | Escenario | Resultado esperado |
|---|---|---|---|
| 1 | `transferMoney` | Transferencia válida | Emisor -200, receptor +200, transacción guardada |
| 2 | `transferMoney` | Saldo insuficiente | Lanza `"Sender Balance not enough"`, no guarda |
| 3 | `transferMoney` | Cuenta emisora inexistente | Lanza `"Sender Account Number not found"` |
| 4 | `transferMoney` | Cuenta receptora inexistente | Lanza `"Receiver Account Number not found"` |
| 5 | `transferMoney` | Cuentas nulas | Lanza `"...cannot be null"` |
| 6 | `getTransactionsForAccount` | Cuenta con historial | Retorna lista con transacciones |
| 7 | `getTransactionsForAccount` | Cuenta sin historial | Retorna lista vacía |

**Ejemplo destacado — Transferencia exitosa:**

```java
@Test
void transferMoney_realizaTransferenciaCorrectamente() {
    // Emisor: 1000, Receptor: 500, Monto: 200
    TransactionDTO result = transactionService.transferMoney(transactionDTO);

    assertEquals(800.0, sender.getBalance());   // 1000 - 200 = 800
    assertEquals(700.0, receiver.getBalance()); // 500  + 200 = 700

    // La transacción DEBE guardarse en BD
    verify(transactionRepository).save(any(Transaction.class));

    // Ambas cuentas DEBEN actualizarse
    verify(customerRepository).save(sender);
    verify(customerRepository).save(receiver);
}
```

**Ejemplo destacado — Saldo insuficiente:**

```java
@Test
void transferMoney_lanzaExcepcionSiSaldoInsuficiente() {
    transactionDTO.setAmount(5000.0); // Más que el saldo de 1000

    IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
            () -> transactionService.transferMoney(transactionDTO));

    assertEquals("Sender Balance not enough", ex.getMessage());

    // Lo más importante: la transacción NO debe guardarse
    verify(transactionRepository, never()).save(any());
}
```

### 3.7 Test de contexto de Spring Boot

Existe además un test de arranque básico que verifica que el contexto de Spring Boot carga correctamente:

```java
@SpringBootTest
class BancoudeaApplicationTests {

    @Test
    void contextLoads() {
        // Verifica que todos los beans se inicializan sin errores
    }
}
```

Este test **sí requiere BD** (usa `@SpringBootTest` que levanta el contexto completo) y detecta errores de configuración como propiedades mal definidas, beans en conflicto o dependencias circulares.

### 3.8 Resultado de ejecución

```
┌─────────────────────────────────┬───────┬────────┐
│ Clase de Test                   │ Tests │ Estado │
├─────────────────────────────────┼───────┼────────┤
│ CustomerServiceTest             │   9   │  PASS  │
│ TransactionServiceTest          │   7   │  PASS  │
│ BancoudeaApplicationTests       │   1   │  PASS  │
├─────────────────────────────────┼───────┼────────┤
│ TOTAL                           │  17   │  PASS  │
└─────────────────────────────────┴───────┴────────┘
Tiempo de ejecución unitarias: ~0.9 segundos (sin BD)
Tiempo contextLoads:           ~3-5 segundos (con BD)
```

---

## 4. Cucumber + Gherkin — Pruebas BDD

### 4.1 ¿Qué es BDD y por qué existe?

**BDD (Behavior-Driven Development)** nació de un problema muy común:

```
Desarrollador: "Terminé la funcionalidad de transferencias"
Cliente:       "Pero cuando el saldo es 0 debería mostrar un mensaje especial"
Desarrollador: "Nadie me dijo eso..."
```

BDD resuelve esto haciendo que **todos** (desarrolladores, testers, clientes, gerentes) escriban y entiendan las pruebas en lenguaje natural.

### 4.2 ¿Qué es Gherkin?

Gherkin es el lenguaje en el que se escriben los escenarios BDD. No es un lenguaje de programación — es lenguaje natural con una estructura específica.

```gherkin
Feature: [Nombre del módulo que se prueba]

  Scenario: [Nombre del caso de uso]
    Given [condición inicial - estado del mundo]
    When  [acción que realiza el usuario o sistema]
    Then  [resultado esperado - lo que debe pasar]
    And   [continúa el Given, When o Then anterior]
```

> **Regla de oro:** Si un cliente de banco puede leer el escenario y entender qué hace el sistema, el Gherkin está bien escrito.

### 4.3 ¿Cómo funciona Cucumber?

Cucumber conecta los archivos `.feature` con el código Java:

```
customer.feature (Gherkin)               CustomerSteps.java (Java)
────────────────────────────             ─────────────────────────────────
Given un cliente guardado con ID 1  ───► @Given("un cliente guardado con ID {long}")
                                         void unClienteGuardadoConID(Long id) {
                                             // aquí va el código real
                                         }
```

### 4.4 Diferencia clave con JUnit: USA LA BD REAL

```
JUnit + Mockito:
Test → Service → Repository (MOCK) → [nada]

Cucumber:
Test → Service → Repository (REAL) → MySQL REAL
```

### 4.5 customer.feature — Análisis de escenarios

```gherkin
Feature: Gestión de clientes

  Scenario: Crear un nuevo cliente exitosamente
    Given un cliente con nombre "Juan", apellido "Perez",
          cuenta "ACC-001" y saldo 1000.0
    When  se crea el cliente
    Then  el cliente es retornado con los mismos datos
```

| # | Escenario | Qué verifica | Tipo |
|---|---|---|---|
| 1 | Crear un nuevo cliente exitosamente | Flujo completo de creación con BD | Happy path |
| 2 | Obtener un cliente por ID existente | Consulta real por ID en MySQL | Happy path |
| 3 | Obtener un cliente por ID inexistente | Manejo correcto de error "no encontrado" | Caso negativo |
| 4 | Eliminar un cliente existente | Eliminación real en MySQL | Happy path |
| 5 | Eliminar un cliente inexistente | No se puede eliminar algo inexistente | Caso negativo |

**Detalle — Escenario 1 (implementación):**

```
@Given → Crea el CustomerDTO en memoria
@When  → Llama customerService.createCustomer() con BD real
@Then  → assertNotNull y assertEquals para cada campo
Limpieza: elimina el registro de BD después del test
```

**Detalle — Escenario 3 (caso negativo):**

```gherkin
Scenario: Obtener un cliente por ID inexistente
  Given que no existe un cliente con ID 99
  When  se busca el cliente con ID 99
  Then  se lanza una excepción con mensaje "Cliente no encontrado"
```

```
Por qué es importante:
  Si el mensaje cambia en el código, este test falla y nos avisa.
  Garantiza consistencia en los mensajes de error de la API.
```

### 4.6 transaction.feature — Análisis de escenarios

```gherkin
Feature: Transferencias bancarias

  Scenario: Transferencia exitosa entre dos cuentas
    Given una cuenta emisora "ACC-001" con saldo 1000.0
    And   una cuenta receptora "ACC-002" con saldo 500.0
    When  se transfiere 200.0 de "ACC-001" a "ACC-002"
    Then  el saldo de "ACC-001" es 800.0
    And   el saldo de "ACC-002" es 700.0
    And   la transacción queda registrada
```

| # | Escenario | Qué verifica | Tipo |
|---|---|---|---|
| 1 | Transferencia exitosa | Saldos actualizados en BD, transacción guardada | Happy path |
| 2 | Transferencia por saldo insuficiente | Bloqueo de la operación, saldos sin cambio | Caso negativo |
| 3 | Cuenta emisora inexistente | Mensaje de error correcto | Caso negativo |
| 4 | Consultar historial de transacciones | Query por cuenta emisora o receptora | Happy path |

**Flujo real que ejecuta el Escenario 1:**

```
1. INSERT en customers (emisor con 1000)
2. INSERT en customers (receptor con 500)
3. Llama TransactionService.transferMoney()
4. UPDATE customers (emisor: balance = 800)
5. UPDATE customers (receptor: balance = 700)
6. INSERT en transactions
7. SELECT para verificar saldo emisor → 800 ✓
8. SELECT para verificar saldo receptor → 700 ✓
9. DELETE transactions (limpieza)
10. DELETE customers x2 (limpieza)
```

### 4.7 Integración con Spring Boot

```java
@CucumberContextConfiguration
@SpringBootTest
public class CucumberSpringConfiguration { }
```

Cada escenario opera sobre la BD real e incluye limpieza automática de datos via `@After` para no contaminar otros tests.

### 4.8 Resultado de ejecución

```
┌──────────────────────────────────┬───────┬────────┐
│ Feature                          │ Tests │ Estado │
├──────────────────────────────────┼───────┼────────┤
│ customer.feature                 │   5   │  PASS  │
│ transaction.feature              │   4   │  PASS  │
├──────────────────────────────────┼───────┼────────┤
│ TOTAL                            │   9   │  PASS  │
└──────────────────────────────────┴───────┴────────┘
Tiempo de ejecución: ~3 segundos
Conexión a BD: REQUERIDA (MySQL)
Reporte HTML: target/cucumber-reports/report.html
```

---

## 5. Karate — Pruebas de API REST

### 5.1 ¿Qué es Karate y en qué se diferencia?

Karate prueba la API como si fuera un cliente externo — hace peticiones HTTP reales contra el servidor desplegado.

```
JUnit y Cucumber:                  Karate:
────────────────                   ────────────────────────────────
Llaman métodos Java                Hace peticiones HTTP reales
Acceden a objetos Java             Solo ve JSON y códigos HTTP
No necesitan servidor              NECESITA el servidor corriendo
Son "caja blanca"                  Es "caja negra" (no ve el código)
```

> **Analogía perfecta:** Si JUnit es el inspector de fábrica que revisa cada pieza, Karate es el cliente que llega a la tienda y prueba el producto terminado.

### 5.2 ¿Por qué Karate no necesita Steps en Java?

```gherkin
# En Cucumber necesitarías escribir un método Java por cada línea.

# En Karate, esto funciona directamente sin Java adicional:
Given path 'customers'
And request { firstName: 'Carlos', balance: 5000.0 }
When method POST
Then status 200
And match response.firstName == 'Carlos'
```

### 5.3 Vocabulario de Karate

| Palabra | Uso |
|---|---|
| `* url 'http://...'` | URL base de la API |
| `* def variable = valor` | Declarar variable |
| `Given path 'ruta'` | Ruta del endpoint |
| `And request { ... }` | Cuerpo de la petición (JSON) |
| `When method POST/GET/PUT/DELETE` | Método HTTP |
| `Then status 200` | Verificar código de respuesta |
| `match response.campo == valor` | Verificar campo del JSON |
| `match response == '#array'` | Verificar que es un arreglo |
| `match response.id == '#notnull'` | Verificar que no es nulo |

### 5.4 customers.feature — Análisis detallado

**Background: configuración compartida**

```gherkin
Background:
  * url 'http://localhost:8080/api'
```

Define la URL base para todos los escenarios del feature — evita repetir la URL en cada uno.

**Escenario 1 — Crear un nuevo cliente:**

```gherkin
Scenario: Crear un nuevo cliente
  Given path 'customers'
  And request { firstName: 'Carlos', lastName: 'Gomez',
                accountNumber: 'KARATE-001', balance: 5000.0 }
  When method POST
  Then status 200
  And match response.firstName == 'Carlos'
  And match response.accountNumber == 'KARATE-001'
  And match response.balance == 5000.0
  And match response.id == '#notnull'
  * def createdId = response.id

  # Limpiar: eliminar el cliente creado
  Given path 'customers/' + createdId
  When method DELETE
  Then status 204
```

```
Flujo HTTP real:
  → POST http://localhost:8080/api/customers
  ← 200 OK { id: 47, firstName: "Carlos", ... }
  → DELETE http://localhost:8080/api/customers/47
  ← 204 No Content

Captura de variables:
  * def createdId = response.id
  Guarda el ID real retornado por la API para usarlo en el DELETE.
  No sabemos de antemano qué ID asignará MySQL.
```

**Escenario 2 — Obtener todos los clientes:**

```gherkin
Scenario: Obtener todos los clientes
  Given path 'customers'
  When method GET
  Then status 200
  And match response == '#array'
```

```
'#array' valida el TIPO, no el contenido específico.
Funciona sin importar cuántos clientes haya en la BD.
```

**Escenarios implementados en customers.feature:**

| # | Escenario | Método HTTP | Validaciones clave |
|---|---|---|---|
| 1 | Crear nuevo cliente | POST | Status 200, campos correctos, ID no nulo |
| 2 | Obtener todos los clientes | GET | Status 200, respuesta es array |
| 3 | Crear y obtener por ID | POST + GET | Status 200, datos coinciden |
| 4 | Actualizar cliente | PUT | Status 200, balance actualizado a 9999 |
| 5 | Eliminar cliente | DELETE | Status 204 |

### 5.5 transactions.feature — Análisis detallado

**Escenario 1 — Transferencia exitosa (el más complejo):**

```gherkin
Scenario: Transferencia exitosa entre dos cuentas
  # Crear emisor
  Given path 'customers'
  And request { accountNumber: 'KAR-SND-01', balance: 1000.0, ... }
  When method POST
  Then status 200
  * def senderId = response.id

  # Crear receptor
  Given path 'customers'
  And request { accountNumber: 'KAR-RCV-01', balance: 500.0, ... }
  When method POST
  Then status 200
  * def receiverId = response.id

  # Transferir
  Given path 'transactions/transfer'
  And request { senderAccountNumber: 'KAR-SND-01',
                receiverAccountNumber: 'KAR-RCV-01',
                amount: 300.0 }
  When method POST
  Then status 200
  And match response.amount == 300.0
  And match response.id == '#notnull'
  And match response.timestamp == '#notnull'

  # Verificar saldo emisor: 1000 - 300 = 700
  Given path 'customers/' + senderId
  When method GET
  Then status 200
  And match response.balance == 700.0

  # Verificar saldo receptor: 500 + 300 = 800
  Given path 'customers/' + receiverId
  When method GET
  Then status 200
  And match response.balance == 800.0
```

```
Este escenario hace 6 requests HTTP en secuencia.
Si cualquier verificación falla, sabemos exactamente en qué paso.
```

**Escenario 2 — Saldo insuficiente:**

```gherkin
  # Emisor tiene 50, intenta transferir 500
  Given path 'transactions/transfer'
  And request { senderAccountNumber: 'KAR-SND-02',
                receiverAccountNumber: 'KAR-RCV-02',
                amount: 500.0 }
  When method POST
  Then status 500
```

> **Nota académica:** En APIs bien diseñadas esto debería ser `400 Bad Request` en lugar de `500 Internal Server Error`, porque es un error del cliente (datos inválidos), no del servidor. Esta es una mejora potencial del proyecto.

**Escenarios implementados en transactions.feature:**

| # | Escenario | Validaciones clave |
|---|---|---|
| 1 | Transferencia exitosa | Saldo emisor -300, receptor +300, timestamp no nulo |
| 2 | Saldo insuficiente | Status 500 |
| 3 | Historial de transacciones | Array con cuenta emisora correcta |

### 5.6 Resultado de ejecución

```
┌──────────────────────────────────┬───────┬──────────────────────────┐
│ Feature                          │ Tests │ Comando                  │
├──────────────────────────────────┼───────┼──────────────────────────┤
│ customers.feature                │   5   │ mvn test                 │
│ transactions.feature             │   3   │ -Dtest=KarateRunnerTest   │
├──────────────────────────────────┼───────┼──────────────────────────┤
│ TOTAL                            │   8   │                          │
└──────────────────────────────────┴───────┴──────────────────────────┘
Requiere: app corriendo en localhost:8080
Reporte:  target/karate-reports/
```

---

## 6. Análisis de Calidad — SonarCloud

### 6.1 ¿Qué es SonarCloud?

SonarCloud es una herramienta de análisis estático de código. Analiza el código fuente **sin ejecutarlo** y detecta:

- **Bugs**: código que puede fallar en tiempo de ejecución
- **Code smells**: código que funciona pero es difícil de mantener
- **Vulnerabilidades**: patrones de seguridad riesgosos
- **Cobertura**: qué porcentaje del código está cubierto por tests

> **Analogía:** Si JUnit, Cucumber y Karate son los inspectores que prueban el puente con carga real, SonarCloud es el arquitecto que revisa los planos antes de construir y señala dónde las vigas están mal calculadas.

### 6.2 Pipeline CI/CD — `.github/workflows/sonarcloud.yml`

El proyecto tiene un workflow de GitHub Actions que ejecuta el análisis automáticamente:

```yaml
name: SonarCloud Analysis

on:
  push:
    branches: [main, develop]
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  sonarcloud:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # necesario para análisis completo de historial

      - name: Set up JDK 17
        uses: actions/setup-java@v3
        with:
          java-version: '17'
          distribution: 'temurin'

      - name: Build with Maven
        run: mvn clean verify -DskipTests

      - name: SonarCloud Scan
        uses: SonarSource/sonarqube-scan-action@v5
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

### 6.3 ¿Cuándo se ejecuta?

| Evento | Acción |
|---|---|
| `git push` a `main` o `develop` | Análisis automático completo |
| Pull Request abierto/actualizado | Análisis del código nuevo con comentarios en el PR |

### 6.4 ¿Qué analiza en este proyecto?

```
BancoUdea/
├── src/main/java/          ← Código de producción analizado
│   ├── controller/         ← ¿Hay endpoints sin validación?
│   ├── service/            ← ¿Hay lógica duplicada?
│   └── repository/         ← ¿Hay queries riesgosas?
└── src/test/java/          ← ¿Los tests tienen buena cobertura?
```

### 6.5 Diferencia con los tests

```
Tests (JUnit/Cucumber/Karate):          SonarCloud:
────────────────────────────            ─────────────────────────────
Prueban comportamiento en runtime       Analiza el código estáticamente
"¿El sistema hace lo que debe?"         "¿El código está bien escrito?"
Requieren ejecutar la app               No requiere ejecutar nada
Detectan bugs funcionales               Detecta bugs potenciales, deuda técnica
```

---

## 7. Comparativa Final

### 7.1 Tabla maestra

| Criterio | JUnit + Mockito | Cucumber + Gherkin | Karate |
|---|---|---|---|
| **Nivel** | Unitario | Integración | API / E2E |
| **Velocidad** | ~1 seg | ~3 seg | ~5-10 seg |
| **BD real** | NO | SI | SI |
| **Servidor HTTP** | NO | NO | SI |
| **Código Java adicional** | Mucho | Medio | Ninguno |
| **Legible por no técnicos** | No | **Sí** | Parcialmente |
| **Detecta** | Bugs de lógica | Bugs de integración | Bugs de contrato HTTP |
| **Tests implementados** | **16** | **9** | **8** |

### 7.2 Cuándo usar cada uno

```
¿Cambié la lógica del servicio?        → JUnit + Mockito primero
¿Cambié la BD o los repositorios?      → Cucumber después
¿Cambié un endpoint o el JSON?         → Karate al final
```

### 7.3 Resumen total del proyecto

```
╔══════════════════════════════════════════════════╗
║         RESUMEN DE TESTING — BANCOUDEA           ║
╠══════════════════════════════════════════════════╣
║  JUnit + Mockito    │  16 tests  │  PASS  ✓      ║
║  Spring Context     │   1 test   │  PASS  ✓      ║
║  Cucumber/Gherkin   │   9 tests  │  PASS  ✓      ║
║  Karate             │   8 tests  │  API*  ✓      ║
╠══════════════════════════════════════════════════╣
║  TOTAL              │  34 tests                  ║
╠══════════════════════════════════════════════════╣
║  * Requiere: mvn spring-boot:run activo          ║
╚══════════════════════════════════════════════════╝
```

### 7.4 Estructura de archivos creados

```
src/test/
├── java/com/udea/bancoudea/
│   ├── BancoudeaApplicationTests.java    ← 1 context load test (Spring)
│   ├── service/
│   │   ├── CustomerServiceTest.java      ← 9 unit tests
│   │   └── TransactionServiceTest.java   ← 7 unit tests
│   ├── cucumber/
│   │   ├── CucumberRunnerTest.java       ← Runner BDD
│   │   ├── CucumberSpringConfiguration  ← Config Spring
│   │   └── steps/
│   │       ├── CustomerSteps.java        ← Steps clientes
│   │       └── TransactionSteps.java     ← Steps transacciones
│   └── karate/
│       ├── KarateRunnerTest.java         ← Runner Karate
│       ├── customers.feature             ← 5 escenarios API
│       └── transactions.feature          ← 3 escenarios API
└── resources/
    └── features/
        ├── customer.feature              ← 5 escenarios BDD
        └── transaction.feature           ← 4 escenarios BDD

.github/
└── workflows/
    └── sonarcloud.yml                    ← CI/CD análisis de calidad
```

---

## 8. Lección Final

La pregunta que siempre surge es: **¿Con cuál me quedo?**

La respuesta es: **con los tres, porque se complementan.**

```
Si solo usas JUnit:
  → Sabes que la lógica funciona pero no sabes si la BD guarda bien.

Si solo usas Cucumber:
  → Sabes que la integración funciona pero tus tests son lentos
    y es difícil aislar exactamente dónde está el bug.

Si solo usas Karate:
  → Sabes que la API responde bien pero si falla no sabes
    si es el Service, el Repository o la BD.

Usando los tres + SonarCloud:
  → Cada herramienta protege una capa específica.
  → Cuando algo falla, el nivel que falla te dice dónde buscar.
  → SonarCloud atrapa deuda técnica antes de que llegue a producción.
  → Los 34 tests juntos dan confianza total en el sistema.
```

### Pirámide aplicada a BancoUdea

```
        Karate (8)
       /──────────\        → Valida el contrato HTTP de la API
      /  Cucumber  \
     /─────────────\       → Valida la integración con MySQL
    /  JUnit (17)   \
   /─────────────────\     → Valida lógica de negocio + contexto Spring

SonarCloud (CI/CD) ──────► Análisis estático continuo en cada push
```

> Entre más abajo en la pirámide, más pruebas hay, más rápidas son y más baratas de mantener. Esta es la distribución ideal para cualquier proyecto de software.

---

*Reporte generado para el proyecto BancoUdea — Spring Boot 3.4.1 / Java 17*
*Stack de testing: JUnit 5 · Mockito · Cucumber 7.20.1 · Karate 1.4.1 · SonarCloud*
